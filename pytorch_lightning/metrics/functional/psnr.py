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
from typing import Optional, Tuple, Union

import torch
from torchmetrics.functional import psnr as _psnr

from pytorch_lightning.metrics.utils import deprecated_metrics, void


@deprecated_metrics(target=_psnr)
def psnr(
    preds: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    base: float = 10.0,
    reduction: str = "elementwise_mean",
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.psnr`. Will be removed in v1.5.0.
    """
    return void(preds, target, data_range, base, reduction, dim)
