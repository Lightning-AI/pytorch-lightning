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
from torchmetrics.functional import confusion_matrix as _confusion_matrix

from pytorch_lightning.metrics.utils import deprecated_metrics, void


@deprecated_metrics(target=_confusion_matrix)
def confusion_matrix(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, normalize: Optional[str] = None, threshold: float = 0.5
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.confusion_matrix`. Will be removed in v1.5.0.
    """
    return void(preds, target, num_classes, normalize, threshold)
