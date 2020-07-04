import torch
from torch.nn import functional as F

from pytorch_lightning.metrics.functional.reduction import reduce


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = None,
    base: float = 10.0,
    reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes the peak signal-to-noise ratio metric

    Args:
        pred: estimated signal
        target: groun truth signal
        data_range: the range of the data. If None, it is determined from the data (max - min).
        base: a base of a logarithm to use (default: 10)
        reduction: method for reducing psnr (default: takes the mean)

    Example:

        >>> from pytorch_lightning.metrics.regression import PSNR
        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> metric = PSNR()
        >>> metric(pred, target)
        tensor(2.5527)
    """

    if data_range is None:
        data_range = max(target.max() - target.min(), pred.max() - pred.min())
    else:
        data_range = torch.tensor(float(data_range))
    mse = F.mse_loss(pred.view(-1), target.view(-1), reduction=reduction)
    psnr_base_e = 2 * torch.log(data_range) - torch.log(mse)
    return psnr_base_e * (10 / torch.log(torch.tensor(base)))
