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
import torch
from typing import Any, Optional, Sequence

from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.metrics.functional.ssim import _ssim_update, _ssim_compute


class SSIM(Metric):
    """
    Computes `Structual Similarity Index Measure
    <https://en.wikipedia.org/wiki/Structural_similarity>`_ (SSIM).

    Args:
        kernel_size: size of the gaussian kernel (default: (11, 11))
        sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03

    Return:
        Tensor with SSIM score

    Example:
        >>> from pytorch_lightning.metrics import SSIM
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim = SSIM()
        >>> ssim(preds, target)
        tensor(0.9219)
    """

    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: str = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        rank_zero_warn(
            'Metric `SSIM` will save all targets and'
            ' predictions in buffer. For large datasets this may lead'
            ' to large memory footprint.'
        )

        self.add_state("y", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _ssim_update(preds, target)
        self.y_pred.append(preds)
        self.y.append(target)

    def compute(self):
        """
        Computes explained variance over state.
        """
        preds = torch.cat(self.y_pred, dim=0)
        target = torch.cat(self.y, dim=0)
        return _ssim_compute(
            preds, target, self.kernel_size, self.sigma, self.reduction, self.data_range, self.k1, self.k2
        )
