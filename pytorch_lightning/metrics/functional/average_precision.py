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
from typing import Optional, Sequence, Tuple, Union, List
import torch

from pytorch_lightning.metrics.functional.precision_recall_curve import (
    _precision_recall_curve_update,
    _precision_recall_curve_compute
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
    output = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        precision, recall, _ = output
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = [ ]
    for (p,r,_) in output:
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
    Compute average precision from prediction scores

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class

    Return:
        Tensor containing average precision score

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> average_precision(x, y)
        tensor(0.3333)
    """
    preds, target, num_classes, pos_label = _average_precision_update(preds, target,
                                                                      num_classes, pos_label)
    return _average_precision_compute(preds, target, num_classes, pos_label, sample_weights)