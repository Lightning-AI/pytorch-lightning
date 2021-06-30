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
from torchmetrics.functional import stat_scores as _stat_scores

from pytorch_lightning.metrics.utils import deprecated_metrics, void


@deprecated_metrics(target=_stat_scores, args_mapping={"is_multiclass": None})
def stat_scores(
    preds: torch.Tensor,
    target: torch.Tensor,
    reduce: str = "micro",
    mdmc_reduce: Optional[str] = None,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.5,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.stat_scores`. Will be removed in v1.5.0.
    """
    return void(preds, target, reduce, mdmc_reduce, num_classes, top_k, threshold, is_multiclass, ignore_index)
