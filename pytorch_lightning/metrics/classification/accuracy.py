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
from pytorch_lightning.metrics.classification.utils import _input_format_classification


class Accuracy(Metric):
    """
    Computes the share of entirely correctly predicted samples.

    This metric generalizes to subset accuracy for multilabel data, and similarly for
    multi-dimensional multi-class data: for the sample to be counted as correct, the the
    class has to be correctly predicted across all extra dimension for each sample of the
    ``N`` dimension. Consider using :class:`HammingLoss` is this is not what you want.

    Accepts logits/probabilities or integer class predictions.

    Forward accepts (see :ref:`Input types` for more information)

    - ``preds`` (float or int tensor): ``(N, ...)``, ``(N, C, ...)`` or ``(N, ..., C)`` where C is the number of classes
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        threshold:
            Threshold probability value for binary or multi-label logits/probabilities. default: 0.5
        logits:
            If predictions are floats, whether the values passed in are probabilities or logits.
            This information is used to know how to transform the ``threshold`` probability before
            comparison.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Example:

        >>> from pytorch_lightning.metrics import Accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)

    """

    def __init__(
        self,
        threshold: float = 0.5,
        logits: bool = True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold
        self.logits = logits

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`Input types` for more information
        on allowed input types.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, _ = _input_format_classification(preds, target, threshold=self.threshold, logits=self.logits)

        N = torch.ones_like(preds[0]).sum()
        extra_dims = list(range(1, len(preds.shape)))
        sample_correct = (preds == target).sum(dim=extra_dims)

        self.correct += (sample_correct == N).sum()
        self.total += preds.shape[0]

    def compute(self) -> float:
        """
        Computes accuracy over state.
        """
        return self.correct.float() / self.total


class HammingLoss(Metric):
    """
    Computes the share of wrongly predicted labels.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label. If this is not what you want, consider using
    :class:`Accuracy`.

    Accepts logits/probabilities or integer class predictions.

    Forward accepts (see :ref:`Input types` for more information)

    - ``preds`` (float or int tensor): ``(N, ...)``, ``(N, C, ...)`` or ``(N, ..., C)`` where C is the number of classes
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        threshold:
            Threshold probability value for binary or multi-label logits/probabilities. default: 0.5
        logits:
            If predictions are floats, whether the values passed in are probabilities or logits.
            This information is used to know how to transform the ``threshold`` probability before
            comparison.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Example:

        >>> from pytorch_lightning.metrics import HammingLoss
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> hamming_loss = HammingLoss()
        >>> hamming_loss(preds, target)
        tensor(0.2500)

    """

    def __init__(
        self,
        threshold: float = 0.5,
        logits: bool = True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold
        self.logits = logits

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`Input types` for more information
        on allowed input types.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, _ = _input_format_classification(preds, target, threshold=self.threshold, logits=self.logits)

        self.correct += (preds == target).sum()
        self.total += preds.numel()

    def compute(self) -> float:
        """
        Computes hamming loss over state.
        """
        return 1 - self.correct.float() / self.total


class TopKAccuracy(Metric):
    """
    For multi-class data, computes the share of correctly predicted samples, where
    a sample is considered correctly predicted if the ground truth label is among
    the ``k`` highest probability labels.

    Forward accepts (see :ref:`Input types` for more information)

    - ``preds`` (float tensor): ``(N, C)``, where C is the number of classes. Whether
        the entries are logits or probabilities is not important, as only their relative
        ranking matters.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        k:
            Number of highest probability predictions considered to find the correct label. Default 2
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Example:

        >>> from pytorch_lightning.metrics import TopKAccuracy
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1, 0, -2], [0.5, -2, 5.0], [-1, 2, 1]])
        >>> accuracy = TopKAccuracy()
        >>> accuracy(preds, target)
        tensor(0.6667)
    """
    def __init__(
        self,
        k: int = 2,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.k = k

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`Input types` for more information
        on allowed input types.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, _ = _input_format_classification(preds, target, logits=True, top_k=self.k)

        self.correct += (preds * target).sum()
        self.total += preds.shape[0]

    def compute(self) -> float:
        """
        Computes top k accuracy over state.
        """
        return self.correct.float() / self.total
