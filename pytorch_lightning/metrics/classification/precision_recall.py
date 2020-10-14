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
import math
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from collections.abc import Mapping, Sequence
from collections import namedtuple

import torch
from torch import nn
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.utils import to_onehot, METRIC_EPS


def _input_format(num_classes: int, preds: torch.Tensor, target: torch.Tensor, threshold=0.5, multilabel=False):
    if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    if len(preds.shape) == len(target.shape) + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    if len(preds.shape) == len(target.shape) and preds.dtype == torch.long and num_classes > 1 and not multilabel:
        # multi-class
        preds = to_onehot(preds, num_classes=num_classes)
        target = to_onehot(target, num_classes=num_classes)

    elif len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
        # binary or multilabel probablities
        preds = (preds >= threshold).long()

    # transpose class as first dim and reshape
    if len(preds.shape) > 1:
        preds = preds.transpose(1, 0)
        target = target.transpose(1, 0)

    return preds.reshape(num_classes, -1), target.reshape(num_classes, -1)


class Precision(Metric):
    """
    Computes the precision metric.

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
        beta: Beta coefficient in the F measure.
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

        >>> from pytorch_lightning.metrics import Precision
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> precision = Precision(num_classes=3)
        >>> precision(preds, target)
        tensor(0.3333)

    """
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = 'micro',
        multilabel: bool = False,
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
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        assert self.average in ('micro', 'macro'), \
            "average passed to the function must be either `micro` or `macro`"

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = _input_format(self.num_classes, preds, target, self.threshold, self.multilabel)

        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += torch.sum(preds * target, dim=1)
        self.predicted_positives += torch.sum(preds, dim=1)

    def compute(self):
        if self.average == 'micro':
            return self.true_positives.sum().float() / (self.predicted_positives.sum() + METRIC_EPS)
        elif self.average == 'macro':
            return (self.true_positives.float() / (self.predicted_positives + METRIC_EPS)).mean()


class Recall(Metric):
    """
    Computes the recall metric.

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
        beta: Beta coefficient in the F measure.
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

        >>> from pytorch_lightning.metrics import Recall
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> recall = Recall(num_classes=3)
        >>> recall(preds, target)
        tensor(0.3333)

    """
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        average: str = 'micro',
        multilabel: bool = False,
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
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        assert self.average in ('micro', 'macro'), \
            "average passed to the function must be either `micro` or `macro`"

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _input_format(self.num_classes, preds, target, self.threshold, self.multilabel)

        # multiply because we are counting (1, 1) pair for true positives
        self.true_positives += torch.sum(preds * target, dim=1)
        self.actual_positives += torch.sum(target, dim=1)

    def compute(self):
        """
        Computes accuracy over state.
        """
        if self.average == 'micro':
            return self.true_positives.sum().float() / (self.actual_positives.sum() + METRIC_EPS)
        elif self.average == 'macro':
            return (self.true_positives.float() / (self.actual_positives + METRIC_EPS)).mean()
