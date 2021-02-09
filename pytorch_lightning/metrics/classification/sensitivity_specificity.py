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

from .precision_recall import Recall
# from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.utils import METRIC_EPS, _input_format_classification_one_hot

__all__ = [
    "Sensitivity",
    "Specificity"
]


class Sensitivity(Recall):
    r"""
    Computes `Sensitivity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_,
    or `TNR (True Positive Rate) <https://en.wikipedia.org/wiki/Confusion_matrix>`_:

    .. math:: \text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively. Different from Recall, sensitivity is a measurement only for
    binary classification. This implementation works with `multi-label` senario as well.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    Args:
        num_classes: Number of classes in the dataset.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5

        average:
            * `'micro'` computes metric globally
            * `'macro'` computes metric for each class and then takes the mean

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import Precision
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> precision = Sensitivity(num_classes=3)
        >>> precision(preds, target)
        tensor(0.3333)

    """
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = 'micro',
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None
    ):
        super().__init__(
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            # multilabel is set to True due to the exact same logic applied to binary and multi-label calculations.
            # TODO: If binary and multi-label calculations are different in the future
            # change this accordingly.
            multilabel=True,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )


class Specificity(Recall):
    r"""
    Computes `Specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_,
    or `TNR (True Negative Rate) <https://en.wikipedia.org/wiki/Confusion_matrix>`_:

    .. math:: \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}

    Where :math:`\text{TN}` and :math:`\text{FP}` represent the number of true negatives and
    false positives respecitively. Similar to Sensitivity, specificity is a measurement only for
    binary classification. This implementation works with `multi-label` senario as well.

    Forward accepts
    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``
    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    Args:
        num_classes: Number of classes in the dataset.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        average:
            * `'micro'` computes metric globally
            * `'macro'` computes metric for each class and then takes the mean
        multilabel: If predictions are from multilabel classification.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:
        >>> from pytorch_lightning.metrics import Specificity
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> specificity = Specificity(num_classes=3)
        >>> specificity(preds, target)
        tensor(0.6667)
    """
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = 'micro',
        pos_labels: Optional[List[int]] = None,
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            # multilabel is set to True due to the exact same logic applied to binary and multi-label calculations.
            # TODO: If binary and multi-label calculations are different in the future
            # change this accordingly.
            multilabel=True,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        preds, target = _input_format_classification_one_hot(
            self.num_classes, preds, target, self.threshold, multilabel=True
        )

        # To reverse the label positive and negatives.
        preds = 1 - preds
        target = 1 - target

        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += torch.sum(preds * target, dim=1)  # calc true negatives actually
        self.actual_positives += torch.sum(target, dim=1)  # calc actual negatives actually
