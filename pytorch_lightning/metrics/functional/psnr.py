from typing import Optional, Tuple, Union

import torch

from pytorch_lightning import utilities
from pytorch_lightning.metrics import utils


def _psnr(
    mean_squared_error: torch.Tensor,
    data_range: float,
    base: float = 10.0,
) -> torch.Tensor:
    psnr_base_e = 2 * torch.log(data_range) - torch.log(mean_squared_error)
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr


def _psnr_compute(psnrs: torch.Tensor, reduction: str = 'elementwise_mean') -> torch.Tensor:
    return utils.reduce(psnrs, reduction)


def _psnr_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    data_range: float,
    base: float = 10.0,
) -> torch.Tensor:
    if dim is None:
        mean_squared_error = torch.mean(torch.pow(preds - target, 2))
    else:
        mean_squared_error = torch.mean(torch.pow(preds - target, 2), dim=dim)
    return _psnr(mean_squared_error, data_range, base=base)


def psnr(
    preds: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    base: float = 10.0,
    reduction: str = 'elementwise_mean',
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> torch.Tensor:
    """
    Computes the peak signal-to-noise ratio

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range:
            the range of the data. If `None`, it is determined from the data (max - min). `data_range` must be given
            when `dim` is not `None`. (default: `None`)
        base: a base of a logarithm to use (default: 10)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        dim:
            Use the mean values of squared errors in the given dimension `dim` to calculate PSNR. If `dim` is a list of
            dimensions, use the mean values over all of them. If `dim` is `None`, use the mean value over all
            dimensions. (default: `None`)
    Return:
        Tensor with PSNR score

    Example:

        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(pred, target)
        tensor(2.5527)

    """
    if dim is None and reduction != 'elementwise_mean':
        utilities.rank_zero_warn(
            f'The `reduction={reduction}` parameter is unused when `dim` is `None` and will not have any effect.'
        )

    if data_range is None:
        if dim is not None:
            # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to calculate
            # `data_range` in the future.
            raise ValueError("`data_range` must be given when `dim` is not `None`.")

        data_range = target.max() - target.min()
    else:
        data_range = torch.tensor(float(data_range))
    psnrs = _psnr_update(preds, target, dim, data_range, base=base)
    return _psnr_compute(psnrs, reduction=reduction)
