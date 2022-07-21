import numpy as np

import itertools
import var
from numpy.random import multivariate_normal
from numpy.linalg import det
from scipy.integrate import odeint
from functools import partial
from itertools import combinations, product


def var1(dim=1, length=1000):
    A_ = np.eye(dim, dim) * 0.45 + np.eye(dim, dim, k=1) * 0.45
    SigmaE = np.eye(dim, dim)

    state_list = np.zeros((length, dim))
    for i in range(1, length):
        state_list[i, :] = A_ @ state_list[i - 1, :]
        state_list[i, :] += multivariate_normal([0] * dim, SigmaE, size=1)[0]
    return state_list


def var4(dim=1, length=1000):
    A_ = np.eye(dim, dim) * 0.45 + np.eye(dim, dim, k=1) * 0.45
    SigmaE = np.eye(dim, dim)
    state_list = np.zeros((length, dim))
    for i in range(4, length):
        state_list[i, :] = A_ @ state_list[i - 4, :]
        state_list[i, :] += -A_ @ state_list[i - 3, :]
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
        state_list[i, 1] += 0.5 * state_list[i - 1, 0] * state_list[i - 1, 1]
        state_list[i, 2] += 0.3 * state_list[i - 1, 1] + 0.5 * state_list[i - 1, 0] ** 2
        state_list[i, :] += 0.4 * multivariate_normal([0] * dim, SigmaE, size=1)[0]
    return state_list


def coupled_henon_maps(dim=3, length=1000, C=0.3):
    state_list = np.zeros((length, dim))
    state_list[0, 0] = 0
    state_list[1, 0] = 0
    for t in range(2, length):
        state_list[t, 0] = 1.4
        state_list[t, 0] -= state_list[t - 1, 0] ** 2
        state_list[t, 0] += 0.3 * state_list[t - 2, 0]
    for t, i in product(range(2, length), range(1, dim)):
        state_list[t, i] = 1.4
        state_list[t, i] -= (
            C * state_list[t - 1, i - 1] + (1 - C) * state_list[t - 1, i]
        ) ** 2
        state_list[t, i] += 0.3 * state_list[t - 2, i]
    return state_list


def coupled_lorenz_system(length=1000, C=3, T=10):
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
    t = np.linspace(0, T, length)
    return odeint(lorenz_df_dt, y0, t)[:, [0, 3, 6]]


def normalize_state_list(state_list):
    return (state_list - np.mean(state_list)) / np.std(state_list)


def load_state_list(param):
    length = 10000
    state_lists = {
        "VAR(1, 1D)": lambda: var1(dim=1, length=length),
        "VAR(1, 2D)": lambda: var1(dim=2, length=length),
        "VAR(1, 3D)": lambda: var1(dim=3, length=length),
        "VAR(4, 1D)": lambda: var4(dim=1, length=length),
        "VAR(4, 2D)": lambda: var4(dim=2, length=length),
        "VAR(4, 3D)": lambda: var4(dim=3, length=length),
        "NLVAR(3, 3D)": lambda: nonlinear_var3(length=length),
        "HENON": lambda: coupled_henon_maps(length=length),
        "LORENZ": lambda: coupled_lorenz_system(length=length),
    }
    st1 = state_lists[param["state_list1"]]()
    st2 = state_lists[param["state_list2"]]()
    state_list = np.concatenate([st1, st2], axis=1)
    return state_list
