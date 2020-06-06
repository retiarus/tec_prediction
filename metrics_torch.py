import torch
from torch import mean
from torch import sqrt


def rms(data, dim=None):
    """Compute RMS."""
    if (dim is None):
        return sqrt(mean(data**2))
    else:
        return sqrt(mean(data**2, dim=dim))
