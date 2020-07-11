import torch

from pytorch_lightning.metrics.functional.regression import (
    mae,
    mse,
    psnr,
    rmse,
    rmsle
)
from pytorch_lightning.metrics.metric import Metric


class MSE(Metric):
    """
    Computes the mean squared loss.

    Example:

        >>> pred = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> metric = MSE()
        >>> metric(pred, target)
        tensor(0.2500)

    """

    def __init__(
            self,
            reduction: str = 'elementwise_mean',
    ):
        """
        Args:
            reduction: a method for reducing mse over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='mse')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the mse loss.
        """
        return mse(pred, target, self.reduction)


class RMSE(Metric):
    """
    Computes the root mean squared loss.

    Example:

        >>> pred = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> metric = RMSE()
        >>> metric(pred, target)
        tensor(0.5000)

    """

    def __init__(
            self,
            reduction: str = 'elementwise_mean',
    ):
        """
        Args:
            reduction: a method for reducing mse over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='rmse')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the rmse loss.
        """
        return rmse(pred, target, self.reduction)


class MAE(Metric):
    """
    Computes the root mean absolute loss or L1-loss.

    Example:

        >>> pred = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> metric = MAE()
        >>> metric(pred, target)
        tensor(0.2500)

    """

    def __init__(
            self,
            reduction: str = 'elementwise_mean',
    ):
        """
        Args:
            reduction: a method for reducing mse over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='mae')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the mae loss.
        """
        return mae(pred, target, self.reduction)


class RMSLE(Metric):
    """
    Computes the root mean squared log loss.

    Example:

        >>> pred = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> metric = RMSLE()
        >>> metric(pred, target)
        tensor(0.0207)

    """

    def __init__(
            self,
            reduction: str = 'elementwise_mean',
    ):
        """
        Args:
            reduction: a method for reducing mse over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='rmsle')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the rmsle loss.
        """
        return rmsle(pred, target, self.reduction)


class PSNR(Metric):
    """
    Computes the peak signal-to-noise ratio

    Example:

        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> metric = PSNR()
        >>> metric(pred, target)
        tensor(2.5527)

    """

    def __init__(
            self,
            data_range: float = None,
            base: int = 10,
            reduction: str = 'elementwise_mean'
    ):
        """
        Args:
            data_range: the range of the data. If None, it is determined from the data (max - min)
            base: a base of a logarithm to use (default: 10)
            reduction: method for reducing psnr (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='psnr')
        self.data_range = data_range
        self.base = float(base)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with psnr score.
        """
        return psnr(pred, target, self.data_range, self.base, self.reduction)
