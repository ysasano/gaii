import torch

import numpy as np
from numpy.linalg import det
from statsmodels.tsa.api import VAR

from scipy.spatial.distance import cdist
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
    return 1 / 2 * np.log(det(SigmaX) / det(var_model.param_u))


def calc_FID(fake_x, real_x):
    fake_mean = np.mean(fake_x)
    fake_cov = np.cov(fake_x)
    real_mean = np.mean(real_x)
    real_cov = np.cov(real_x)
    return cdist(fake_mean, real_mean) + np.trace(
        fake_cov + real_cov - 2 * sqrtm(fake_cov @ real_cov)
    )
