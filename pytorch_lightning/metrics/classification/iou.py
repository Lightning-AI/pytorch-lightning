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
from typing import Any, Optional
import torch
from pytorch_lightning.metrics.functional.reduction import reduce
from pytorch_lightning.metrics.functional.confusion_matrix import _confusion_matrix_compute
from pytorch_lightning.metrics.functional.iou import _iou_from_confmat
from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix


class IoU(ConfusionMatrix):
    """
    Intersection over union, or Jaccard index calculation. Works with binary, multiclass, and multilabel data.
    Accepts logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. Has no effect if given an int that is not in the
            range [0, num_classes-1], where num_classes is either given or derived from pred and target. By default, no
            index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of the class index were present in
            `pred` AND no instances of the class index were present in `target`. For example, if we have 3 classes,
            [0, 0] for `pred`, and [0, 2] for `target`, then class 1 would be assigned the `absent_score`. Default is
            0.0.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:
        >>> from pytorch_lightning.metrics import IoU
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> iou = IoU(num_classes=2)
        >>> iou(pred, target)
        tensor(0.9660)

    """

    def __init__(
            self,
            num_classes: int,
            ignore_index: Optional[int] = None,
            absent_score: float = 0.0,
            threshold: float = 0.5,
            reduction: str = 'elementwise_mean',
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            normalize=None,
            threshold=threshold,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.absent_score = absent_score

    def compute(self) -> torch.Tensor:
        """
        Computes iou
        """
        return _iou_from_confmat(self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction)
