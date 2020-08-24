# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import torch

from pytorch_lightning.metrics.functional.regression import (
    mae,
    mse,
    psnr,
    rmse,
    rmsle,
    ssim
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
            reduction: a method to reduce metric score over labels (default: takes the mean)
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
            reduction: a method to reduce metric score over labels (default: takes the mean)
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
    Computes the mean absolute loss or L1-loss.

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
            reduction: a method to reduce metric score over labels (default: takes the mean)
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
            reduction: a method to reduce metric score over labels (default: takes the mean)
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
            reduction: a method to reduce metric score over labels (default: takes the mean)
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


class SSIM(Metric):
    """
    Computes Structual Similarity Index Measure

    Example:

        >>> pred = torch.rand([16, 1, 16, 16])
        >>> target = pred * 0.75
        >>> metric = SSIM()
        >>> metric(pred, target)
        tensor(0.9219)

    """

    def __init__(
            self,
            kernel_size: Sequence[int] = (11, 11),
            sigma: Sequence[float] = (1.5, 1.5),
            reduction: str = "elementwise_mean",
            data_range: float = None,
            k1: float = 0.01,
            k2: float = 0.03
    ):
        """
        Args:
            kernel_size: Size of the gaussian kernel (default: (11, 11))
            sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass away
                - sum: add elements

            data_range: Range of the image. If ``None``, it is determined from the image (max - min)
            k1: Parameter of SSIM. Default: 0.01
            k2: Parameter of SSIM. Default: 0.03
        """
        super().__init__(name="ssim")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: Estimated image
            target: Ground truth image

        Return:
            A Tensor with SSIM score.
        """
        return ssim(pred, target, self.kernel_size, self.sigma, self.reduction, self.data_range, self.k1, self.k2)
