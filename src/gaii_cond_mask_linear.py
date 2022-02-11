import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utility import from_torch, to_torch, tril

cuda = torch.cuda.is_available()
rng = np.random.default_rng()


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = mask.detach()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return F.linear(input, self.weight * self.mask)

    def extra_repr(self):
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features
        )


class Generator(nn.Module):
    def __init__(self, N, mask):
        super(Generator, self).__init__()
        assert mask.shape == (N, N)
        self.fc1 = MaskedLinear(N, N, mask.detach())

        # reparametrization (PRML 演習問題11.5)
        # lower triangular mask
        mask2 = tril(N)
        self.fc2 = MaskedLinear(N, N, mask2)

    def forward(self, x, z):
        if cuda:
            x = x.cuda()
            z = z.cuda()
        return self.fc1(x) + self.fc2(z.detach())


class Discriminator(nn.Module):
    def __init__(self, N):
        super(Discriminator, self).__init__()
        if N <= 16:
            self.fc1 = nn.Linear(N * 2, 16)
            self.fc2 = nn.Linear(16, 1)
        elif N <= 64:
            self.fc1 = nn.Linear(N * 2, 32)
            self.fc2 = nn.Linear(32, 1)
        else:
            self.fc1 = nn.Linear(N * 2, 64)
            self.fc2 = nn.Linear(64, 1)

    def forward(self, x, y):
        if cuda:
            x = x.cuda()
            y = y.cuda()
        x_y = torch.hstack((x, y))
        x_y = torch.tanh(self.fc1(x_y))
        return torch.sigmoid(self.fc2(x_y))


def sample_xy(batch_size, state_list):
    idx = rng.choice(len(state_list) - 1, batch_size)
    xs = to_torch(state_list[idx])
    ys = to_torch(state_list[idx + 1])
    return [xs, ys]


def sample_z(batch_size, N):
    return torch.randn(batch_size, N)


def fit_q(state_list, mask, batch_size=200, n_step=2000, debug=False):
    # mode = "GAN"
    mode = "f-GAN:KL"
    N = mask.shape[0]
    cuda = torch.cuda.is_available()
    if cuda:
        mask = to_torch(mask).cuda()
    else:
        mask = to_torch(mask)

    G = Generator(N, mask)
    D = Discriminator(N)
    adversarial_loss = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3)
    if cuda:
        G.cuda()
        D.cuda()
        adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
        fake_x, _ = sample_xy(batch_size, state_list)
        z = sample_z(batch_size, N)
        fake_y = G(fake_x, z)

        # real xとreal yの生成
        real_x, real_y = sample_xy(batch_size, state_list)

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
        fake_x, _ = sample_xy(batch_size, state_list)
        z = sample_z(batch_size, N)
        fake_y = G(fake_x, z)

        # real xとreal yの生成
        real_x, real_y = sample_xy(batch_size, state_list)

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

        if i % 1000 == 0 and debug:
            w = from_torch(G.state_dict()["fc1.weight"] * mask)
            print(w)
            L = from_torch(G.state_dict()["fc2.weight"]) * from_torch(tril(N))
            print(L @ L.T)

        if i % 100 == 0:
            failure_check.append((i, d_score.item(), g_score.item()))

    return {
        "G": G,
        "D": D,
        "N": N,
        "batch_size": batch_size,
        "mask": mask,
        "state_list": state_list,
        "failure_check": pd.DataFrame(
            failure_check, columns=["i", "d_score", "g_score"]
        ).set_index("i"),
        "js": js,
    }
