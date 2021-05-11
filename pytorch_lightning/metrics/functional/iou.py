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
from torchmetrics.functional import iou as _iou

from pytorch_lightning.metrics.utils import deprecated_metrics
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_3, _TORCHMETRICS_LOWER_THAN_0_3


@deprecated_metrics(target=_iou, skip_if=_TORCHMETRICS_GREATER_EQUAL_0_3)
@deprecated_metrics(target=_iou, args_mapping=dict(pred="preds"), skip_if=_TORCHMETRICS_LOWER_THAN_0_3)
def iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.iou`. Will be removed in v1.5.0.
    """
