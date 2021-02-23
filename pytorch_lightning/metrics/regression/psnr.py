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
from typing import Any, Optional, Sequence, Tuple, Union

import torch

from pytorch_lightning import utilities
from pytorch_lightning.metrics.functional.psnr import _psnr_compute, _psnr_update
from pytorch_lightning.metrics.metric import Metric


class PSNR(Metric):
    r"""
    Computes `peak signal-to-noise ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_ (PSNR):

    .. math:: \text{PSNR}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)}\right)

    Where :math:`\text{MSE}` denotes the `mean-squared-error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ function.

    Args:
        data_range:
            the range of the data. If None, it is determined from the data (max - min).
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use (default: 10)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import PSNR
        >>> psnr = PSNR()
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)

    """

    def __init__(
        self,
        data_range: Optional[float] = None,
        base: float = 10.0,
        reduction: str = 'elementwise_mean',
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        if dim is None and reduction != 'elementwise_mean':
            utilities.rank_zero_warn(f'The `reduction={reduction}` will not have any effect when `dim` is None.')

        if dim is None:
            self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[])
            self.add_state("total", default=[])

        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError("The `data_range` must be given when `dim` is not None.")

            self.data_range = None
            self.add_state("min_target", default=torch.tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=torch.tensor(0.0), dist_reduce_fx=torch.max)
        else:
            self.register_buffer("data_range", torch.tensor(float(data_range)))
        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, n_obs = _psnr_update(preds, target, dim=self.dim)
        if self.dim is None:
            if self.data_range is None:
                # keep track of min and max target values
                self.min_target = min(target.min(), self.min_target)
                self.max_target = max(target.max(), self.max_target)

            self.sum_squared_error += sum_squared_error
            self.total += n_obs
        else:
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(n_obs)

    def compute(self):
        """
        Compute peak signal-to-noise ratio over state.
        """
        if self.data_range is not None:
            data_range = self.data_range
        else:
            data_range = self.max_target - self.min_target

        if self.dim is None:
            sum_squared_error = self.sum_squared_error
            total = self.total
        else:
            sum_squared_error = torch.cat([values.flatten() for values in self.sum_squared_error])
            total = torch.cat([values.flatten() for values in self.total])
        return _psnr_compute(sum_squared_error, total, data_range, base=self.base, reduction=self.reduction)
