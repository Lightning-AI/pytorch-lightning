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
from typing import Optional, Sequence

import torch
from torchmetrics.functional import ssim as _ssim

from pytorch_lightning.metrics.utils import deprecated_metrics, void


@deprecated_metrics(target=_ssim)
def ssim(
    preds: torch.Tensor,
    target: torch.Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.ssim`. Will be removed in v1.5.0.
    """
    return void(preds, target, kernel_size, sigma, reduction, data_range, k1, k2)
