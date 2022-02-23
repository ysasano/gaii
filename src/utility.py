import torch

import numpy as np
from numpy.linalg import det


def to_torch(x):
    return torch.from_numpy(x.astype(np.float32)).clone()


def from_torch(x):
    return x.to("cpu").detach().numpy().copy()


def tril(N):
    return to_torch(np.tril(np.ones((N, N))))


def calc_MI(SigmaX, SigmaE):
    return 1 / 2 * np.log(det(SigmaX) / det(SigmaE))
