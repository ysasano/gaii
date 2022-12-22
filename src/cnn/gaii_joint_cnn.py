import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

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
            nn.Linear(latent_size, 128),  # => 128
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            Reshape(2, 8, 8),  # => 2 x 8 x 8
            nn.ConvTranspose2d(
                2, 2, kernel_size=4, stride=2, padding=1, bias=False
            ),  # => 2 x 16 x 16
            # nn.BatchNorm2d(2),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                2, 1, kernel_size=1, stride=1, padding=0, bias=False
            ),  # => 1 x 16 x 16
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
            batch_size, self.length, 2, 1, 16, 16
        )  # => length x 2 x 1 x 16 x 16
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
                1, 2, kernel_size=1, stride=1, padding=0, bias=False
            ),  # => 2 x 16 x 16
            nn.BatchNorm2d(2),
            nn.LeakyReLU(alpha),
            nn.Conv2d(
                2, 2, kernel_size=4, stride=2, padding=1, bias=False
            ),  # => 2 x 8 x 8
            nn.LeakyReLU(alpha),
            nn.Flatten(),
            nn.Linear(128, 100),  # => 100
        )

        self.linear1 = nn.Linear(self.size, self.size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(self.size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        # time distribute encode
        x = x.reshape(batch_size * self.length * 2, 1, 16, 16)  # => 1 x 16 x 16
        x = self.encoder(x)  # => 100
        x = x.reshape(batch_size, self.length * 2 * 100)  # => length x 2 x 100

        # joint_dense
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        return self.sigmoid(self.linear2(x))


def sample_x(batch_size, state_list, length):
    stream_size = state_list.shape[0]
    stream_length = state_list.shape[1]

    # stream_size個の動画からランダムにbatch_size個サンプリング
    stream_idxes = rng.choice(stream_size, batch_size)
    # stream_size x stream_length x 2 x 1 x w x h => batch_size x stream_length x 2 x 1 x w x h
    seq = state_list[stream_idxes, ...]

    # それぞれのサンプルについて、stream_length長の動画からランダムにlength長の区間を抽出
    begin_idx = rng.choice(stream_length - length - 1, batch_size)
    idx_span = np.array([np.arange(b, b + length) for b in begin_idx])
    # batch_size x stream_length x 2 x 1 x w x h => batch_size x length x 2 x 1 x w x h
    seq = np.stack([seq[i, idx_span[i], ...] for i in range(batch_size)], axis=0)
    return to_torch(seq)


def sample_z(batch_size, length):
    return torch.randn(batch_size, length, 200)  # => length x 200


def save_gif(images, i):
    # images: batch_size x length x 2 x 1 x w x h
    images = (images + 1) / 2 * 255
    size = images.shape[4]

    images1 = images[0, :, 0, 0, :, :]
    images2 = images[0, :, 1, 0, :, :]
    images_cat = np.concatenate((images1, images2), axis=2)
    images_flat = np.reshape(images_cat, (-1, size, size * 2))

    images_pil = []
    for image in images_flat:
        images_pil.append(Image.fromarray(image).convert("P"))
    images_pil[0].save(
        "data/generate_image_{}.gif".format(i),
        save_all=True,
        append_images=images_pil[1:],
        optimize=False,
        duration=40,
        loop=0,
    )


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
    # real_label = torch.ones(batch_size, 1, requires_grad=False) * 0.7
    # fake_label = torch.ones(batch_size, 1, requires_grad=False) * 0.3

    FID_all = []
    loss_all = []
    js_all = []
    d_loss_std_all = []
    grad_norm_all = []
    failure_check = []
    js_ema = None
    d_loss_ema = None
    f_star = lambda t: torch.exp(t - 1)
    state_list = torch.stack(
        [to_torch(images1), to_torch(images2)], dim=2
    )  # stream_size x stream_length x w x h => stream_size x stream_length x 2 x w x h
    state_list = torch.unsqueeze(
        state_list, 3
    )  # => stream_size x stream_length x 2 x 1 x w x h

    for i in range(n_step):
        # print(i, n_step)
        # ====================
        # Discriminatorの学習
        # ====================
        d_optimizer.zero_grad()

        # fake xの生成
        z1 = sample_z(batch_size, length)  # => length x 200
        z2 = sample_z(batch_size, length)  # => length x 200
        fake_x = G(z1, z2)
        # real xの生成
        real_x = sample_x(batch_size, state_list, length)  # => length x 2 x 1 x w x h

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
        real_x = sample_x(batch_size, state_list, length)  # => length x 2 x 1 x w x h
        if i % 100 == 0:
            torch.set_printoptions(precision=1, sci_mode=False, linewidth=200)
            print(real_x[0, 0, 0, 0, :, :])
            # print(real_x[0, 1, 0, 0, :, :])
            print(fake_x[0, 0, 0, 0, :, :].detach())
            # print(fake_x[0, 1, 0, 0, :, :].detach())
            torch.set_printoptions(profile="default")

        # Discriminatorを騙すように学習
        D_fake = D(fake_x)
        D_real = D(real_x)
        if mode == "GAN":
            fake_loss = adversarial_loss(D_fake, real_label)
            real_loss = adversarial_loss(D_real, fake_label)
            # print(fake_loss)
            # print(real_loss)
            g_loss = fake_loss + real_loss
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

        # JSのEMA
        if js_ema is None:
            js_ema = js
        else:
            alpha = 0.001
            js_ema = alpha * js + (1 - alpha) * js_ema

        # Dlossの分散
        if d_loss_ema is None:
            d_loss_ema = d_loss.item()
            d_loss_std = 0
        else:
            alpha = 0.001
            d_loss_ema = alpha * d_loss.item() + (1 - alpha) * d_loss_ema
            d_loss_std = (
                alpha * (d_loss.item() - d_loss_ema) ** 2 + (1 - alpha) * d_loss_std
            )

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
            d_loss_std_all.append((i, d_loss_std))

            # 勾配のノルム
            grad_norm = 0
            for d in D.parameters():
                param_norm = d.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            for g in G.parameters():
                param_norm = g.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1.0 / 2)
            grad_norm_all.append((i, grad_norm))

        if i % 1000 == 0:
            save_gif(fake_x.detach().numpy(), i)

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
        "d_loss_std_all": pd.DataFrame(
            d_loss_std_all, columns=["i", "d_loss_std_all"]
        ).set_index("i"),
        "grad_norm_all": pd.DataFrame(
            grad_norm_all, columns=["i", "grad_norm_all"]
        ).set_index("i"),
        "js": js_ema,
    }
