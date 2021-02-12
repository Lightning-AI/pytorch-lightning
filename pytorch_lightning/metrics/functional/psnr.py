from typing import Optional, Tuple

import torch

from pytorch_lightning.utilities import rank_zero_warn


def _psnr_compute(
    sum_squared_error: torch.Tensor,
    n_obs: int,
    data_range: float,
    base: float = 10.0,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    if reduction != 'elementwise_mean':
        rank_zero_warn(f'The `reduction={reduction}` parameter is unused and will not have any effect.')
    psnr_base_e = 2 * torch.log(data_range) - torch.log(sum_squared_error / n_obs)
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr


def _psnr_update(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
    sum_squared_error = torch.sum(torch.pow(preds - target, 2))
    n_obs = target.numel()
    return sum_squared_error, n_obs


def psnr(
    preds: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    base: float = 10.0,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    Computes the peak signal-to-noise ratio

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range: the range of the data. If None, it is determined from the data (max - min)
        base: a base of a logarithm to use (default: 10)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with PSNR score

    Example:

        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(pred, target)
        tensor(2.5527)

    """
    if data_range is None:
        data_range = target.max() - target.min()
    else:
        data_range = torch.tensor(float(data_range))
    sum_squared_error, n_obs = _psnr_update(preds, target)
    return _psnr_compute(sum_squared_error, n_obs, data_range, base, reduction)
