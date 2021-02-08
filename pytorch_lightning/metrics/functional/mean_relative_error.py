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
from typing import Tuple

import torch

from pytorch_lightning.metrics.utils import _check_same_shape


def _mean_relative_error_update(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
    _check_same_shape(preds, target)
    target_nz = target.clone()
    target_nz[target == 0] = 1
    sum_rltv_error = torch.sum(torch.abs((preds - target) / target_nz))
    n_obs = target.numel()
    return sum_rltv_error, n_obs


def _mean_relative_error_compute(sum_rltv_error: torch.Tensor, n_obs: int) -> torch.Tensor:
    return sum_rltv_error / n_obs


def mean_relative_error(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes mean relative error

    Args:
        pred: estimated labels
        target: ground truth labels

    Return:
        Tensor with mean relative error

    Example:

        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_relative_error(x, y)
        tensor(0.1250)

    """
    sum_rltv_error, n_obs = _mean_relative_error_update(preds, target)
    return _mean_relative_error_compute(sum_rltv_error, n_obs)
