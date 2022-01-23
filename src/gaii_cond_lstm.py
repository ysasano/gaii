import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view

from utility import from_torch, to_torch, tril

cuda = torch.cuda.is_available()
rng = np.random.default_rng()


def get_invert_permutation(permutation):
    permutation = np.array(permutation)
    invert_permutation = np.empty(permutation.size, dtype=permutation.dtype)
    for i in np.arange(permutation.size):
        invert_permutation[permutation[i]] = i
    return invert_permutation


class Generator(nn.Module):
    def __init__(self, partation, hidden_size):
        super(Generator, self).__init__()
        self.partation = partation
        self.invert_partation = get_invert_permutation(np.concatenate(partation))
        self.lstm1 = nn.LSTM(
            len(partation[0]) + hidden_size, len(partation[0]), num_layers=2
        )
        self.lstm2 = nn.LSTM(
            len(partation[1]) + hidden_size, len(partation[1]), num_layers=2
        )

    def forward(self, x, z):
        x_z = torch.cat((x[:, :, self.partation[0]], z), dim=-1)
        _, (_, cn) = self.lstm1(x_z)
        hidden1 = torch.squeeze(cn[-1], 0)

        x_z = torch.cat((x[:, :, self.partation[1]], z), dim=-1)
        _, (_, cn) = self.lstm2(x_z)
        hidden2 = torch.squeeze(cn[-1], 0)

        hidden = torch.cat((hidden1, hidden2), dim=-1)
        hidden = hidden[:, self.invert_partation]

        return torch.unsqueeze(hidden, 0)


class Discriminator(nn.Module):
    def __init__(self, size):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(size, size, num_layers=2)
        self.fc_out = nn.Linear(size, 1)
        self.size = size

    def forward(self, x, y):
        x_y = torch.cat((x, y), dim=0)
        _, (_, cn) = self.lstm(x_y)
        hidden = torch.squeeze(cn[-1], 0)
        return self.fc_out(hidden)


def sample_xy(batch_size, state_list, length):
    length_with_y = length + 1
    idx = rng.choice(len(state_list) - length_with_y - 1, batch_size)
    idx_span = np.array([np.arange(i, i + length_with_y) for i in idx])
    seq = to_torch(state_list[idx_span, :])
    seq = torch.permute(seq, (1, 0, 2))
    return seq[:-1, :, :], torch.unsqueeze(seq[-1, :, :], 0)


def sample_z(batch_size, N, length):
    z = torch.randn(batch_size, N)
    z = torch.reshape(z, (1, batch_size, N))
    return z.tile(length, 1, 1)


def fit_q(state_list, partation, batch_size=200, n_step=2000, length=20, debug=False):
    # mode = "GAN"
    mode = "f-GAN:KL"
    N = sum(len(p) for p in partation)

    G = Generator(partation, N)
    D = Discriminator(N)
    adversarial_loss = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3)
    if debug:
        print(G)
        print(D)

    real_label = torch.ones(batch_size, 1, requires_grad=False)
    fake_label = torch.zeros(batch_size, 1, requires_grad=False)

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
        "js": js,
    }
