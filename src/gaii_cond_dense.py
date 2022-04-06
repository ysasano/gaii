import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utility import (
    calc_FID,
    from_torch,
    to_torch,
    get_invert_permutation,
    normalize_state_list,
)

cuda = torch.cuda.is_available()
rng = np.random.default_rng()


class Generator(nn.Module):
    def __init__(self, length, partation, use_time_invariant_term):
        super(Generator, self).__init__()
        self.partation = partation
        self.invert_partation = get_invert_permutation(np.concatenate(partation))
        self.use_time_invariant_term = use_time_invariant_term

        z1_size = len(partation[0]) * length
        z2_size = len(partation[1]) * length

        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z1_size, z1_size),
            nn.ReLU(),
            nn.Linear(z1_size, z1_size),
            nn.Unflatten(-1, (length, len(partation[0]))),
        )

        self.seq2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z2_size, z2_size),
            nn.ReLU(),
            nn.Linear(z2_size, z2_size),
            nn.Unflatten(-1, (length, len(partation[1]))),
        )

        self.depth = len(partation[0]) + len(partation[1])
        self.linear_corr = nn.Linear(self.depth, self.depth)

    def forward(self, x, z):
        x = normalize_state_list(x)
        x1 = self.seq1(x[:, :, self.partation[0]])
        x2 = self.seq2(x[:, :, self.partation[1]])
        corr = self.linear_corr(z)

        hidden = torch.cat((x1, x2), dim=-1)
        hidden = hidden[:, :, self.invert_partation]
        if self.use_time_invariant_term:
            hidden += corr

        return hidden.mean(dim=1, keepdim=True)


class Discriminator(nn.Module):
    def __init__(self, length, dim):
        super(Discriminator, self).__init__()
        self.activation = nn.ReLU()
        self.size = length * dim
        self.linear1 = nn.Linear(self.size, self.size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(self.size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        xy = normalize_state_list(xy)
        xy = xy.view(-1, self.size)
        xy = self.activation(self.linear1(xy))
        xy = self.dropout(xy)
        xy = self.sigmoid(self.linear2(xy))
        return xy


def sample_xy(batch_size, state_list, length):
    idx = rng.choice(len(state_list) - length, batch_size)
    idx_span = np.array([np.arange(i, i + length) for i in idx])
    seq = to_torch(state_list[idx_span, :])
    return seq[:, :-1, :], torch.unsqueeze(seq[:, -1, :], 1)


def sample_z(batch_size, N, length):
    z = torch.randn(batch_size * (length - 1), N)
    return z.view(batch_size, length - 1, N)


def fit_q(
    state_list,
    partation,
    batch_size=800,
    n_step=20000,
    length=4,
    use_time_invariant_term=False,
    debug=False,
):
    mode = "GAN"
    # mode = "f-GAN:KL"
    N = sum(len(p) for p in partation)

    G = Generator(length - 1, partation, use_time_invariant_term)
    D = Discriminator(length, N)
    adversarial_loss = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=1e-4)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-4)
    if debug:
        print(G)
        print(D)
    if cuda:
        G.cuda()
        D.cuda()

    real_label = torch.ones(batch_size, 1, requires_grad=False)
    fake_label = torch.zeros(batch_size, 1, requires_grad=False)

    FID_all = []
    loss_all = []
    js_all = []
    failure_check = []
    f_star = lambda t: torch.exp(t - 1)

    for i in range(n_step):
        # ====================
        # Discriminatorの学習
        # ====================
        d_optimizer.zero_grad()

        # fake xとfake yの生成
        fake_x, _ = sample_xy(batch_size, state_list, length)
        z = sample_z(batch_size, N, length)
        fake_y = G(fake_x, z)

        # real xとreal yの生成
        real_x, real_y = sample_xy(batch_size, state_list, length)

        # リアルのサンプルとニセのサンプルを正しく見分けられるように学習
        D_fake = D(fake_x, fake_y.detach())
        D_real = D(real_x, real_y)
        if mode == "GAN":
            fake_loss = adversarial_loss(D_fake, fake_label)
            real_loss = adversarial_loss(D_real, real_label)
            d_loss = real_loss + fake_loss
        elif mode == "f-GAN:KL":
            fake_loss = f_star(D_fake).mean()
            real_loss = -D_real.mean()
            d_loss = real_loss + fake_loss

        d_loss.backward()
        d_optimizer.step()

        # ====================
        # Generatorの学習
        # ====================
        g_optimizer.zero_grad()

        # fake xとfake yの生成
        fake_x, _ = sample_xy(batch_size, state_list, length)
        z = sample_z(batch_size, N, length)
        fake_y = G(fake_x, z)

        # real xとreal yの生成
        real_x, real_y = sample_xy(batch_size, state_list, length)

        # Discriminatorを騙すように学習
        D_fake = D(fake_x, fake_y)
        D_real = D(real_x, real_y)
        if mode == "GAN":
            fake_loss = adversarial_loss(D_fake, real_label)
            real_loss = adversarial_loss(D_real, fake_label)
            g_loss = real_loss + fake_loss
        elif mode == "f-GAN:KL":
            fake_loss = -f_star(D_fake).mean()
            real_loss = D_real.mean()
            g_loss = real_loss + fake_loss

        g_loss.backward()
        g_optimizer.step()

        if mode == "GAN":
            js = (-d_loss.item() + 2 * math.log(2)) / 2
        elif mode == "f-GAN:KL":
            js = -d_loss.item()

        # 崩壊モードチェック
        g_score = D_fake.mean()
        d_score = 1 / 2 * D_real.mean() + 1 / 2 * (1 - D_fake.mean())

        if i % 100 == 0 and debug:
            print(
                "[Count %d/%d]    [JS: %f] [G loss: %f] [D loss: %f]"
                % (i, n_step, js, g_loss.item(), d_loss.item())
            )

        if i % 100 == 0:
            failure_check.append((i, d_score.item(), g_score.item()))
            FID_all.append((i, calc_FID(from_torch(fake_x), from_torch(real_x))))
            js_all.append((i, js))
            loss_all.append((i, d_loss.item(), g_loss.item()))

    return {
        "G": G,
        "D": D,
        "N": N,
        "batch_size": batch_size,
        "partation": partation,
        "length": length,
        "state_list": state_list,
        "failure_check": pd.DataFrame(
            failure_check, columns=["i", "d_score", "g_score"]
        ).set_index("i"),
        "FID_all": pd.DataFrame(FID_all, columns=["i", "FID"]).set_index("i"),
        "js_all": pd.DataFrame(js_all, columns=["i", "js"]).set_index("i"),
        "loss_all": pd.DataFrame(loss_all, columns=["i", "g_loss", "d_loss"]).set_index(
            "i"
        ),
        "js": js,
    }
