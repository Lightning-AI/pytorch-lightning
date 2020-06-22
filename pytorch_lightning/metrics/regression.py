import torch.nn.functional as F
import torch
from pytorch_lightning.metrics.metric import Metric

__all__ = ['MSE', 'RMSE', 'MAE', 'RMSLE']


class MSE(Metric):
    """
    Computes the mean squared loss.
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


        Example:

            >>> pred = torch.tensor([0., 1, 2, 3])
            >>> target = torch.tensor([0., 1, 2, 2])
            >>> metric = MSE()
            >>> metric(pred, target)
            tensor(0.2500)

        """
        super().__init__(name='mse')
        if reduction == 'elementwise_mean':
            reduction = 'mean'
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
        return F.mse_loss(pred, target, self.reduction)


class RMSE(Metric):
    """
    Computes the root mean squared loss.
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


        Example:

            >>> pred = torch.tensor([0., 1, 2, 3])
            >>> target = torch.tensor([0., 1, 2, 2])
            >>> metric = RMSE()
            >>> metric(pred, target)
            tensor(0.5000)

        """
        super().__init__(name='rmse')
        if reduction == 'elementwise_mean':
            reduction = 'mean'
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
        return torch.sqrt(F.mse_loss(pred, target, self.reduction))


class MAE(Metric):
    """
    Computes the root mean absolute loss or L1-loss.
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


        Example:

            >>> pred = torch.tensor([0., 1, 2, 3])
            >>> target = torch.tensor([0., 1, 2, 2])
            >>> metric = MAE()
            >>> metric(pred, target)
            tensor(0.2500)

        """
        super().__init__(name='mae')
        if reduction == 'elementwise_mean':
            reduction = 'mean'
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
        return F.l1_loss(pred, target, self.reduction)


class RMSLE(Metric):
    """
    Computes the root mean squared log loss.
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


        Example:

            >>> pred = torch.tensor([0., 1, 2, 3])
            >>> target = torch.tensor([0., 1, 2, 2])
            >>> metric = RMSLE()
            >>> metric(pred, target)
            tensor(0.0207)

        """
        super().__init__(name='rmsle')
        if reduction == 'elementwise_mean':
            reduction = 'mean'
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
        return F.mse_loss(torch.log(pred + 1), torch.log(target + 1), self.reduction)
