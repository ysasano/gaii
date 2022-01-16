import torch

import numpy as np

def to_torch(x):
    return torch.from_numpy(x.astype(np.float32)).clone()

def from_torch(x):
    return x.to('cpu').detach().numpy().copy()

def tril(N):
    return to_torch(np.tril(np.ones((N,N))))