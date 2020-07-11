import torch
from torch.nn import functional as F

from pytorch_lightning.metrics.functional.reduction import reduce


def mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes mean squared error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing mse (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with MSE

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mse(x, y)
        tensor(0.2500)

    """
    mse = F.mse_loss(pred, target, reduction='none')
    mse = reduce(mse, reduction=reduction)
    return mse


def rmse(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes root mean squared error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing rmse (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with RMSE


        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> rmse(x, y)
        tensor(0.5000)

    """
    rmse = torch.sqrt(mse(pred, target, reduction=reduction))
    return rmse


def mae(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes mean absolute error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing mae (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with MAE

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mae(x, y)
        tensor(0.2500)

    """
    mae = F.l1_loss(pred, target, reduction='none')
    mae = reduce(mae, reduction=reduction)
    return mae


def rmsle(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes root mean squared log error

    Args:
        pred: estimated labels
        target: ground truth labels
        reduction: method for reducing rmsle (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with RMSLE

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> rmsle(x, y)
        tensor(0.0207)

    """
    rmsle = mse(torch.log(pred + 1), torch.log(target + 1), reduction=reduction)
    return rmsle


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = None,
    base: float = 10.0,
    reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes the peak signal-to-noise ratio

    Args:
        pred: estimated signal
        target: groun truth signal
        data_range: the range of the data. If None, it is determined from the data (max - min)
        base: a base of a logarithm to use (default: 10)
        reduction: method for reducing psnr (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum add elements

    Return:
        Tensor with PSNR score

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

    mse_score = mse(pred.view(-1), target.view(-1), reduction=reduction)
    psnr_base_e = 2 * torch.log(data_range) - torch.log(mse_score)
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr
