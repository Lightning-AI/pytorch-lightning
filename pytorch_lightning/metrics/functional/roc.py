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
from typing import Optional, Sequence, Tuple, List, Union

import torch

from pytorch_lightning.metrics.functional.precision_recall_curve import (
    _precision_recall_curve_update,
    _binary_clf_curve
)


def _roc_update(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    return _precision_recall_curve_update(preds, target, num_classes, pos_label)


def _roc_compute(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        pos_label: int,
        sample_weights: Optional[Sequence] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:

    if num_classes == 1:
        fps, tps, thresholds = _binary_clf_curve(
            preds=preds,
            target=target,
            sample_weights=sample_weights,
            pos_label=pos_label
        )
        # Add an extra threshold position
        # to make sure that the curve starts at (0, 0)
        tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
        fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
        thresholds = torch.cat([thresholds[0][None] + 1, thresholds])

        if fps[-1] <= 0:
            raise ValueError("No negative samples in targets, false positive value should be meaningless")
        fpr = fps / fps[-1]

        if tps[-1] <= 0:
            raise ValueError("No positive samples in targets, true positive value should be meaningless")
        tpr = tps / tps[-1]

        return fpr, tpr, thresholds

    # Recursively call per class
    roc_vals = []
    for c in range(num_classes):
        preds_c = preds[:, c]
        res = roc(
            preds=preds_c,
            target=target,
            num_classes=1,
            pos_label=c,
            sample_weights=sample_weights,
        )
        roc_vals.append(res)

    return roc_vals


def roc(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        sample_weights: Optional[Sequence] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    preds, target, num_classes, pos_label = _roc_update(preds, target,
                                                        num_classes, pos_label)
    return _roc_compute(preds, target, num_classes, pos_label, sample_weights)
