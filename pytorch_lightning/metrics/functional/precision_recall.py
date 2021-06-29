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
from torch import Tensor
from torchmetrics.functional import precision as _precision
from torchmetrics.functional import precision_recall as _precision_recall
from torchmetrics.functional import recall as _recall

from pytorch_lightning.metrics.utils import deprecated_metrics, void, _TORCHMETRICS_GREATER_EQUAL_0_4, _TORCHMETRICS_LOWER_THAN_0_4


@deprecated_metrics(target=_precision, skip_if=_TORCHMETRICS_GREATER_EQUAL_0_4)
@deprecated_metrics(target=_precision, args_mapping={"multilabel": None, "is_multiclass": "multiclass"}, skip_if=_TORCHMETRICS_LOWER_THAN_0_4)
def precision(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.precision`. Will be removed in v1.5.0.
    """
    return void(preds, target, average, mdmc_average, ignore_index, num_classes, threshold, top_k, is_multiclass)


@deprecated_metrics(target=_recall, skip_if=_TORCHMETRICS_GREATER_EQUAL_0_4)
@deprecated_metrics(target=_recall, args_mapping={"multilabel": None, "is_multiclass": "multiclass"}, skip_if=_TORCHMETRICS_LOWER_THAN_0_4)
def recall(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.accuracy`. Will be removed in v1.5.0.
    """
    return void(preds, target, average, mdmc_average, ignore_index, num_classes, threshold, top_k, is_multiclass)


@deprecated_metrics(target=_precision_recall, skip_if=_TORCHMETRICS_GREATER_EQUAL_0_4)
@deprecated_metrics(target=_precision_recall, args_mapping={"multilabel": None, "is_multiclass": "multiclass"}, skip_if=_TORCHMETRICS_LOWER_THAN_0_4)
def precision_recall(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.precision_recall`. Will be removed in v1.5.0.
    """
    return void(preds, target, average, mdmc_average, ignore_index, num_classes, threshold, top_k, is_multiclass)
