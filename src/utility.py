import torch

import numpy as np
from numpy.linalg import det
from statsmodels.tsa.api import VAR

from scipy.linalg import sqrtm


def to_torch(x):
    return torch.from_numpy(x.astype(np.float32)).clone()


def from_torch(x):
    return x.to("cpu").detach().numpy().copy()


def tril(N):
    return to_torch(np.tril(np.ones((N, N))))


def calc_MI(state_list):
    var_model = VAR(state_list).fit(trend="n", maxlags=1)
    SigmaX = np.cov(state_list.T)
    return 1 / 2 * np.log(det(SigmaX) / det(var_model.sigma_u))


def calc_FID(fake_x, real_x):
    dim = fake_x.shape[1] * fake_x.shape[2]
    fake_mean = np.mean(fake_x.reshape((-1, dim)), axis=0)
    fake_cov = np.cov(
        fake_x.reshape((-1, dim)).T,
    )
    real_mean = np.mean(real_x.reshape((-1, dim)), axis=0)
    real_cov = np.cov(real_x.reshape((-1, dim)).T)
    return (fake_mean - real_mean) @ (fake_mean - real_mean) + np.trace(
        fake_cov + real_cov - 2 * sqrtm(fake_cov @ real_cov)
    )


def get_invert_permutation(permutation):
    permutation = np.array(permutation)
    invert_permutation = np.empty(permutation.size, dtype=permutation.dtype)
    for i in np.arange(permutation.size):
        invert_permutation[permutation[i]] = i
    return invert_permutation
