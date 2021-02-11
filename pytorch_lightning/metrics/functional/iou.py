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

from pytorch_lightning.metrics.functional.confusion_matrix import _confusion_matrix_update
from pytorch_lightning.metrics.functional.reduction import reduce
from pytorch_lightning.metrics.utils import get_num_classes


def _iou_from_confmat(
    confmat: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    reduction: str = 'elementwise_mean',
):
    intersection = torch.diag(confmat)
    union = confmat.sum(0) + confmat.sum(1) - intersection

    # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
    scores = intersection.float() / union.float()
    scores[union == 0] = absent_score

    # Remove the ignored class index from the scores.
    if ignore_index is not None and ignore_index >= 0 and ignore_index < num_classes:
        scores = torch.cat([
            scores[:ignore_index],
            scores[ignore_index + 1:],
        ])
    return reduce(scores, reduction=reduction)


def iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    r"""
    Computes `Intersection over union, or Jaccard index calculation <https://en.wikipedia.org/wiki/Jaccard_index>`_:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Where: :math:`A` and :math:`B` are both tensors of the same size,
    containing integer class values. They may be subject to conversion from
    input data (see description below).

    Note that it is different from box IoU.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If pred has an extra dimension as in the case of multi-class scores we
    perform an argmax on ``dim=1``.

    Args:
        preds: tensor containing predictions from model (probabilities, or labels) with shape ``[N, d1, d2, ...]``
        target: tensor containing ground truth labels with shape ``[N, d1, d2, ...]``
        ignore_index: optional int specifying a target class to ignore. If given,
            this class index does not contribute to the returned score, regardless
            of reduction method. Has no effect if given an int that is not in the
            range [0, num_classes-1], where num_classes is either given or derived
            from pred and target. By default, no index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of
            the class index were present in `pred` AND no instances of the class
            index were present in `target`. For example, if we have 3 classes,
            [0, 0] for `pred`, and [0, 2] for `target`, then class 1 would be
            assigned the `absent_score`.
        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5
        num_classes:
            Optionally specify the number of classes
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        IoU score : Tensor containing single value if reduction is
        'elementwise_mean', or number of classes if reduction is 'none'

    Example:

        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> iou(pred, target)
        tensor(0.9660)

    """

    num_classes = get_num_classes(pred=pred, target=target, num_classes=num_classes)
    confmat = _confusion_matrix_update(pred, target, num_classes, threshold)
    return _iou_from_confmat(confmat, num_classes, ignore_index, absent_score, reduction)
