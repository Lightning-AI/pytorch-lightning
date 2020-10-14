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

from pytorch_lightning.metrics.classification.accuracy import _input_format
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_warn


class ConfusionMatrix(Metric):
    """
    Computes the confusion matrix. Works with binary, multiclass, and multilabel data.
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
        normalize: normalization mode for confusion matrix. Default is None,
            meaning no normalization. Choose between normalization over the
            targets (`'true`', most commonly used), the predictions (`'pred`') or
            over hole matrix (`'all'`)
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import ConfusionMatrix
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = ConfusionMatrix(num_classes=2)
        >>> confmat(preds, target)
        tensor([[2., 0.],
                [1., 1.]])

    """
    def __init__(
        self,
        num_classes: int,
        normalize: Optional[str] = None,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.num_classes = num_classes
        self.normalize = normalize
        self.threshold = threshold

        allowed_normalize = ('true', 'pred', 'all', None)
        assert self.normalize in allowed_normalize, \
            f"Argument average needs to one of the following: {allowed_normalize}"

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _input_format(preds, target, self.threshold)

        unique_mapping = (target.view(-1) * self.num_classes + preds.view(-1)).to(torch.long)
        bins = torch.bincount(unique_mapping, minlength=self.num_classes ** 2)

        self.confmat += bins.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Computes confusion matrix
        """
        if self.normalize is not None:
            if self.normalize == 'true':
                cm = self.confmat / self.confmat.sum(axis=1, keepdim=True)
            elif self.normalize == 'pred':
                cm = self.confmat / self.confmat.sum(axis=0, keepdim=True)
            elif self.normalize == 'all':
                cm = self.confmat / self.confmat.sum()
            nan_elements = cm[torch.isnan(cm)].nelement()
            if nan_elements != 0:
                cm[torch.isnan(cm)] = 0
                rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
            return cm
        return self.confmat
