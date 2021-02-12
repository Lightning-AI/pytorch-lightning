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
from typing import List, Optional, Sequence, Tuple, Union

import torch

from pytorch_lightning.metrics.functional.precision_recall_curve import (
    _precision_recall_curve_compute,
    _precision_recall_curve_update,
)


def _average_precision_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    return _precision_recall_curve_update(preds, target, num_classes, pos_label)


def _average_precision_compute(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    pos_label: int,
    sample_weights: Optional[Sequence] = None
) -> Union[List[torch.Tensor], torch.Tensor]:
    precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = []
    for p, r in zip(precision, recall):
        res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))
    return res


def average_precision(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    sample_weights: Optional[Sequence] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Computes the average precision score.

    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
        sample_weights: sample weights for each data point

    Returns:
        tensor with average precision. If multiclass will return list
        of such tensors, one for each class

    Example (binary case):

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision(pred, target, pos_label=1)
        tensor(1.)

    Example (multiclass case):

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision(pred, target, num_classes=5)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]

    """
    preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes, pos_label)
    return _average_precision_compute(preds, target, num_classes, pos_label, sample_weights)
