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

        # https://kikaben.com/dcgan-mnist/
        self.seq1 = nn.Sequential(
            nn.Flatten(),  # => length * 100
            nn.Linear(length * 100, 512),  # => 1024
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            Reshape(8, 8, 8),  # => 8 x 8 x 8
            nn.ConvTranspose2d(
                8, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32,
                length,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),  # => length x 64 x 64
            Reshape(1, length, 64, 64),
            nn.Sigmoid(),
        )

        self.seq2 = nn.Sequential(
            nn.Flatten(),  # => length * 100
            nn.Linear(length * 100, 512),  # => 784
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            Reshape(8, 8, 8),  # => 16 x 8 x 8
            nn.ConvTranspose2d(
                8, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32,
                length,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),  # => length x 64 x 64
            Reshape(1, length, 64, 64),
            nn.Sigmoid(),
        )

        self.corr = nn.Sequential(
            nn.Flatten(),  # => 2 * 100
            nn.Linear(200, 512),  # => 784
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            Reshape(8, 8, 8),  # => 16 x 8 x 8
            nn.ConvTranspose2d(
                8, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),  # => 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32,
                2,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),  # => 2 x 64 x 64
            Reshape(2, 1, 64, 64),  # => 2 x 1 x 64 x 64
            nn.Sigmoid(),
        )

    def forward(self, z1, z2):
        z1_1 = z1[:, 0, :, :]  # 2 x length x 100 => length x 100
        z1_2 = z1[:, 1, :, :]  # 2 x length x 100 => length x 100
        x1 = self.seq1(z1_1)  # => 1 x length x 64 x 64
        x2 = self.seq2(z1_2)  # => 1 x length x 64 x 64
        hidden = torch.cat((x1, x2), dim=1)  # => 2 x length x 64 x 64

        corr_list = []
        for t in range(self.length):
            z_t = z2[:, :, t, :]  # 2 x length x 100 => 2 x 100
            corr = self.corr(z_t)  # => 2 x 1 x 64 x 64
            corr_list.append(corr)
        hidden_corr = torch.cat(corr_list, dim=2)  # => 2 x length x 64 x 64

        if self.use_time_invariant_term:
            hidden += hidden_corr

        return hidden


class Discriminator(nn.Module):
    def __init__(self, length):
        super(Discriminator, self).__init__()
        alpha = 0.01
        # 2 x length x 64 x 64 => 16 x 32 x 32
        self.conv1 = nn.Sequential(
            Reshape(2 * length, 64, 64),
            nn.Conv2d(2 * length, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(alpha),
        )

        # 16 x 32 x 32 => 16 x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(alpha),
        )

        # 16 x 16 x 16 => 16 x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(alpha),
        )

        # 16 x 8 x 8 => 1024
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # => 16 x 32 x 32
        x = self.conv2(x)  # => 16 x 16 x 16
        x = self.conv3(x)  # => 16 x 8 x 8
        x = self.fc(x)
        return self.sigmoid(x)


def sample_x(batch_size, state_list, length):
    stream_num = state_list.shape[0]
    stream_idxes = rng.choice(stream_num, batch_size)
    # all_batch_size x 2 x all_length x w x h => batch_size x 2 x all_length x width x height
    seq = state_list[stream_idxes, :, :, :, :]

    idx = rng.choice(state_list.shape[2] - length - 1, batch_size)
    idx_span = np.array([np.arange(i, i + length) for i in idx])
    # batch_size x 2 x all_length x w x h => batch_size x 2 x length x width x height
    seq = np.stack([seq[i, :, idx_span[i], :, :] for i in range(batch_size)], axis=0)
    return to_torch(seq)


def sample_z(batch_size, length):
    z = torch.randn(batch_size * length, 2, 100)
    return z.view(batch_size, 2, length, 100)  # => 2 x length x 100


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
    js_ema = None
    f_star = lambda t: torch.exp(t - 1)
    state_list = torch.stack(
        [to_torch(images1), to_torch(images2)], dim=1
    )  # all_length x width x height => 2 x all_length x width x height

    for i in range(n_step):
        print(i, n_step)
        # ====================
        # Discriminatorの学習
        # ====================
        d_optimizer.zero_grad()

        # fake xの生成
        z1 = sample_z(batch_size, length)  # => 2 x length x 100
        z2 = sample_z(batch_size, length)  # => 2 x length x 100
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
        z1 = sample_z(batch_size, length)  # => 2 x length x 100
        z2 = sample_z(batch_size, length)  # => 2 x length x 100
        fake_x = G(z1, z2)

        # real xの生成
        real_x = sample_x(
            batch_size, state_list, length
        )  # => 2 x length x width x height

        # Discriminatorを騙すように学習
        D_fake = D(fake_x)
        D_real = D(real_x)
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
        "loss_all": pd.DataFrame(loss_all, columns=["i", "g_loss", "d_loss"]).set_index(
            "i"
        ),
        "js": js_ema,
    }
