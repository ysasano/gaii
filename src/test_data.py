import numpy as np

import itertools
import var
from numpy.random import multivariate_normal
from numpy.linalg import det
from scipy.integrate import odeint
from functools import partial
from itertools import combinations, product


def var1(dim=1, length=1000):

    A_ = np.eye(dim, dim) * 0.2 + np.eye(dim, dim, k=1) * 0.2
    SigmaE = np.eye(dim, dim)

    state_list = np.zeros((length, dim))
    for i in range(1, length):
        state_list[i, :] = A_ @ state_list[i - 3, :]
        state_list[i, :] += multivariate_normal([0] * dim, SigmaE, size=1)[0]
    return state_list


def var4(dim=1, length=1000):
    A_ = np.eye(dim, dim) * 0.2 + np.eye(dim, dim, k=1) * 0.2
    SigmaE = np.eye(dim, dim)
    state_list = np.zeros((length, dim))
    for i in range(3, length):
        state_list[i, :] = A_ @ state_list[i - 3, :]
        state_list[i, :] += -A_ @ state_list[i - 2, :]
        state_list[i, :] += multivariate_normal([0] * dim, SigmaE, size=1)[0]
    return state_list


def nonlinear_var3(length=1000):
    dim = 3  # モデルで固定
    SigmaE = np.eye(dim, dim)
    state_list = np.zeros((length, dim))
    for i in range(1, length):
        state_list[i, :] = (
            3.4
            * state_list[i - 1, :]
            * (1 - state_list[i - 1, :] ** 2)
            * np.exp(-state_list[i - 1, :] ** 2)
        )
        state_list[i, 1] += 0.5 * state_list[i - 1, 0] * state_list[i - 2, 0]
        state_list[i, 2] += 0.3 * state_list[i - 1, 0] + 0.5 * state_list[i - 2, 0] ** 2
        state_list[i, :] += 0.4 * multivariate_normal([0] * dim, SigmaE, size=1)[0]
    return state_list


def coupled_henon_maps(dim=1, length=1000, C=0.3):
    state_list = np.zeros((dim, length))
    state_list[0, 0] = 0
    state_list[0, 1] = 0
    for t in range(2, length):
        state_list[0, t] = 1.4
        state_list[0, t] -= state_list[0, t - 1] ** 2
        state_list[0, t] -= 0.3 * state_list[0, t - 2]
    for i, t in product(range(1, dim), range(2, length)):
        state_list[i, t] = 1.4
        state_list[i, t] -= C * state_list[i - 1, t - 1]
        state_list[i, t] -= (1 - C) * state_list[i, t - 1]
        state_list[i, t] += 0.3 * state_list[i, t - 2]
    return state_list


def coupled_lorenz_system(length=1000, C=3):
    def lorenz_df_dt(state, _):
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = state
        return [
            -10 * x1 + 10 * y1,
            -x1 * z1 + 28 * x1 - y1,
            x1 * y1 - 8 / 3 * z1,
            -10 * x2 + 10 * y2 + C * (x1 - x2),
            -x2 * z2 + 28 * x2 - y2,
            x2 * y2 - 8 / 3 * z2,
            -10 * x3 + 10 * y3 + C * (x2 - x3),
            -x3 * z3 + 28 * x3 - y3,
            x3 * y3 - 8 / 3 * z3,
        ]

    y0 = [0.01, 0, 0, 0.02, 0, 0, 0.03, 0, 0]
    t = np.linspace(0, 1, length)
    return odeint(lorenz_df_dt, y0, t)[:, [0, 3, 6]]


def generate(length=1000):
    state_lists = [
        var1(dim=1, length=length),
        var1(dim=2, length=length),
        var1(dim=3, length=length),
        var4(dim=1, length=length),
        var4(dim=2, length=length),
        var4(dim=3, length=length),
        nonlinear_var3(length=length),
        coupled_lorenz_system(length=length),
    ]

    for st1, st2 in combinations(state_lists, 2):
        yield np.concatenate(st1, st2)
