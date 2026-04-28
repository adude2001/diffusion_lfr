from collections import namedtuple
import torch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(x):
    return x


def prob_mask_like(shape, prob, device):
    if prob <= 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    if prob >= 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
