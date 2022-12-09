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
)

cuda = torch.cuda.is_available()
rng = np.random.default_rng()


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()

        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class Generator(nn.Module):
    def __init__(self, length, use_time_invariant_term):
        super(Generator, self).__init__()
        self.use_time_invariant_term = use_time_invariant_term
        self.length = length
        latent_size = 100

        # https://kikaben.com/dcgan-mnist/
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),  # => 1024
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            Reshape(8, 8, 8),  # => 8 x 8 x 8
            nn.ConvTranspose2d(
                8, 4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 4 x 16 x 16
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                4, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 1 x 32 x 32
        )

        z1_size = length * latent_size
        z2_size = length * latent_size

        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z1_size, z1_size),
            nn.LeakyReLU(0.01),
            nn.Linear(z1_size, z1_size),
            nn.Unflatten(-1, (length, latent_size)),
        )

        self.seq2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z2_size, z2_size),
            nn.LeakyReLU(0.01),
            nn.Linear(z2_size, z2_size),
            nn.Unflatten(-1, (length, latent_size)),
        )

        self.linear_corr = nn.Linear(latent_size * 2, latent_size * 2)
        self.tanh = nn.Tanh()

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        latent_size = z1.shape[-1] // 2

        # joint_dense
        x1 = self.seq1(z1[:, :, :latent_size])  # length x 200 => length x 100
        x2 = self.seq2(z1[:, :, -latent_size:])  # length x 200 => length x 100
        corr = self.linear_corr(z2)  # length x 200 => length x 200

        hidden = torch.cat((x1, x2), dim=-1)  # => length x 200
        if self.use_time_invariant_term:
            hidden += corr  # => length x 200

        # time distribute decode
        hidden = hidden.reshape(batch_size * self.length * 2, latent_size)  # => 100
        hidden = self.decoder(hidden)
        hidden = hidden.reshape(
            batch_size, self.length, 2, 1, 32, 32
        )  # => length x 2 x 1 x 32 x 32
        return self.tanh(hidden)


class Discriminator(nn.Module):
    def __init__(self, length):
        super(Discriminator, self).__init__()
        alpha = 0.01
        self.length = length
        self.size = length * 2 * 100
        self.activation = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 2, kernel_size=5, stride=2, padding=2, bias=False
            ),  # => 2 x 16 x 16
            nn.LeakyReLU(alpha),
            nn.Conv2d(
                2, 4, kernel_size=5, stride=2, padding=2, bias=False
            ),  # => 4 x 8 x 8
            nn.LeakyReLU(alpha),
            nn.Flatten(),
            nn.Linear(256, 100),  # => 100
        )

        self.linear1 = nn.Linear(self.size, self.size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(self.size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        # time distribute encode
        x = x.reshape(batch_size * self.length * 2, 1, 32, 32)  # => 1 x 32 x 32
        x = self.encoder(x)  # => 100
        x = x.reshape(batch_size, self.length * 2 * 100)  # => length x 2 x 100

        # joint_dense
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        return self.sigmoid(self.linear2(x))


def sample_x(batch_size, state_list, length):
    stream_num = state_list.shape[0]
    stream_idxes = rng.choice(stream_num, batch_size)
    # all_batch_size x all_length x 2 x 1 x w x h => batch_size x all_length x 2 x 1 x width x height
    seq = state_list[stream_idxes, ...]

    idx = rng.choice(state_list.shape[1] - length - 1, batch_size)
    idx_span = np.array([np.arange(i, i + length) for i in idx])
    # batch_size x all_length x 2 x 1 x w x h => batch_size x length x 2 x 1 x w x h
    seq = np.stack([seq[i, idx_span[i], ...] for i in range(batch_size)], axis=0)
    return to_torch(seq)


def sample_z(batch_size, length):
    z = torch.randn(batch_size * length, 200)
    return z.view(batch_size, length, 200)  # => length x 200


def fit_q(
    images1,
    images2,
    batch_size=32,
    n_step=20000,
    length=4,
    use_time_invariant_term=False,
    debug=False,
):
    mode = "GAN"
    # mode = "f-GAN:KL"
    G = Generator(length, use_time_invariant_term)
    D = Discriminator(length)
    adversarial_loss = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=1e-5)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-5)
    if debug:
        print(G)
        print(D)
    if cuda:
        G.cuda()
        D.cuda()

    real_label = torch.ones(batch_size, 1, requires_grad=False) * 0.7
    fake_label = torch.ones(batch_size, 1, requires_grad=False) * 0.3

    FID_all = []
    loss_all = []
    js_all = []
    failure_check = []
    js_ema = None
    f_star = lambda t: torch.exp(t - 1)
    state_list = torch.stack(
        [to_torch(images1), to_torch(images2)], dim=2
    )  # all_length x width x height => all_length x 2 x width x height
    state_list = torch.reshape(
        state_list,
        [
            state_list.shape[0],
            state_list.shape[1],
            state_list.shape[2],
            1,
            state_list.shape[3],
            state_list.shape[4],
        ],
    )  # => all_length x 2 x 1 x width x height

    for i in range(n_step):
        print(i, n_step)
        # ====================
        # Discriminatorの学習
        # ====================
        d_optimizer.zero_grad()

        # fake xの生成
        z1 = sample_z(batch_size, length)  # => length x 200
        z2 = sample_z(batch_size, length)  # => length x 200
        fake_x = G(z1, z2)
        # real xの生成
        real_x = sample_x(
            batch_size, state_list, length
        )  # => 2 x length x width x height

        # リアルのサンプルとニセのサンプルを正しく見分けられるように学習
        D_fake = D(fake_x.detach())
        D_real = D(real_x)
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

        # fake xの生成
        z1 = sample_z(batch_size, length)  # => length x 200
        z2 = sample_z(batch_size, length)  # => length x 200
        fake_x = G(z1, z2)

        # real xの生成
        real_x = sample_x(
            batch_size, state_list, length
        )  # => length x 2 x 1 x width x height
        # print(real_x[0, 0, 0, 0, :, :])
        # print(fake_x[0, 0, 0, 0, :, :])

        # Discriminatorを騙すように学習
        D_fake = D(fake_x)
        D_real = D(real_x)
        # print(D_real[:5])
        # print(D_fake[:5])
        if mode == "GAN":
            fake_loss = adversarial_loss(D_fake, real_label)
            real_loss = adversarial_loss(D_real, fake_label)
            # print(fake_loss)
            # print(real_loss)
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

        if js_ema is None:
            js_ema = js
        else:
            alpha = 0.001
            js_ema = alpha * js + (1 - alpha) * js_ema

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
            js_all.append((i, js, js_ema))
            loss_all.append((i, d_loss.item(), g_loss.item()))

    return {
        "G": G,
        "D": D,
        "batch_size": batch_size,
        "length": length,
        "failure_check": pd.DataFrame(
            failure_check, columns=["i", "d_score", "g_score"]
        ).set_index("i"),
        "FID_all": pd.DataFrame(FID_all, columns=["i", "FID"]).set_index("i"),
        "js_all": pd.DataFrame(js_all, columns=["i", "js", "js_ema"]).set_index("i"),
        "loss_all": pd.DataFrame(loss_all, columns=["i", "d_loss", "g_loss"]).set_index(
            "i"
        ),
        "js": js_ema,
    }
