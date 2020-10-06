import torch

from typing import Any, Callable, Optional, Union


def dim_zero_cat(x):
    return torch.cat(x, dim=0)


def dim_zero_sum(x):
    return torch.sum(x, dim=0)


def dim_zero_mean(x):
    return torch.mean(x, dim=0)


def _flatten(x):
    return [item for sublist in x for item in sublist]
