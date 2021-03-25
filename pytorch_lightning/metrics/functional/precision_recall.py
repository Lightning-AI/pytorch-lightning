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
from typing import Optional

import torch
from torchmetrics.functional import precision as _precision
from torchmetrics.functional import precision_recall as _precision_recall
from torchmetrics.functional import recall as _recall

from pytorch_lightning.utilities.deprecation import deprecated


@deprecated(target=_precision, ver_deprecate="1.3.0", ver_remove="1.5.0")
def precision(
    preds: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.precision`. Will be removed in v1.5.0.
    """


@deprecated(target=_recall, ver_deprecate="1.3.0", ver_remove="1.5.0")
def recall(
    preds: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.accuracy`. Will be removed in v1.5.0.
    """


@deprecated(target=_precision_recall, ver_deprecate="1.3.0", ver_remove="1.5.0")
def precision_recall(
    preds: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.precision_recall`. Will be removed in v1.5.0.
    """
