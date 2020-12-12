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

from pytorch_lightning.metrics.functional.f_beta import (
    _fbeta_update,
    _fbeta_compute
)
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_warn


class FBeta(Metric):
    r"""
    Computes `F-score <https://en.wikipedia.org/wiki/F-score>`_, specifically:

    .. math::
        F_\beta = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    Where :math:`\beta` is some positive real factor. Works with binary, multiclass, and multilabel data.
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
        beta: Beta coefficient in the F measure.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5

        average:
            - ``'micro'`` computes metric globally
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``'none'`` computes and returns the metric per class

        multilabel: If predictions are from multilabel classification.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import FBeta
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f_beta = FBeta(num_classes=3, beta=0.5)
        >>> f_beta(preds, target)
        tensor(0.3333)

    """

    def __init__(
        self,
        num_classes: int,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group,
        )

        self.num_classes = num_classes
        self.beta = beta
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        allowed_average = ("micro", "macro", "weighted", None)
        if self.average not in allowed_average:
            raise ValueError('Argument `average` expected to be one of the following:'
                             f' {allowed_average} but got {self.average}')

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        true_positives, predicted_positives, actual_positives = _fbeta_update(
            preds, target, self.num_classes, self.threshold, self.multilabel
        )

        self.true_positives += true_positives
        self.predicted_positives += predicted_positives
        self.actual_positives += actual_positives

    def compute(self) -> torch.Tensor:
        """
        Computes fbeta over state.
        """
        return _fbeta_compute(self.true_positives, self.predicted_positives,
                              self.actual_positives, self.beta, self.average)


# todo: remove in v1.2
class Fbeta(FBeta):
    r"""
    Computes `F-score <https://en.wikipedia.org/wiki/F-score>`_

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.classification.f_beta.FBeta`
    """
    def __init__(
        self,
        num_classes: int,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        rank_zero_warn(
            "This `Fbeta` was deprecated in v1.0.x in favor of"
            " `from pytorch_lightning.metrics.classification.f_beta import FBeta`."
            " It will be removed in v1.2.0", DeprecationWarning
        )
        super().__init__(
            num_classes, beta, threshold, average, multilabel, compute_on_step, dist_sync_on_step, process_group
        )


class F1(FBeta):
    """
    Computes F1 metric. F1 metrics correspond to a harmonic mean of the
    precision and recall scores.

    Works with binary, multiclass, and multilabel data.
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
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5

        average:
            - ``'micro'`` computes metric globally
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``'none'`` computes and returns the metric per class

        multilabel: If predictions are from multilabel classification.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:
        >>> from pytorch_lightning.metrics import F1
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1 = F1(num_classes=3)
        >>> f1(preds, target)
        tensor(0.3333)
    """

    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            beta=1.0,
            threshold=threshold,
            average=average,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
