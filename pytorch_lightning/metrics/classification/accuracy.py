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
from typing import Any, Callable, Optional

import torch
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.classification.utils import _input_format_classification


class Accuracy(Metric):
    """
    Computes the share of entirely correctly predicted samples.

    This metric generalizes to subset accuracy for multilabel data, and similarly for
    multi-dimensional multi-class data: for the sample to be counted as correct, the the
    class has to be correctly predicted across all extra dimension for each sample in the
    ``N`` dimension. Consider using :class:`~pytorch_lightning.metrics.classification.HammingLoss`
    is this is not what you want.

    Accepts all input types listed in :ref:`metrics:Input types`.

    Args:
        threshold:
            Threshold probability value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. If ``logits=True``,
            this value is transformed to logits by ``logit_t = ln(t / (1-t))``. Default: 0.5
        logits:
            If predictions are floats, whether they are probabilities or logits. Default ``True``.
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
        Update state with predictions and targets. See :ref:`metrics:Input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, logits, or labels)
            target: Ground truth values
        """
        preds, target, _ = _input_format_classification(preds, target, threshold=self.threshold, logits=self.logits)

        extra_dims = list(range(1, len(preds.shape)))
        sample_correct = (preds == target).sum(dim=extra_dims)

        self.correct += (sample_correct == preds[0].numel()).sum()
        self.total += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        return self.correct.float() / self.total


class HammingLoss(Metric):
    """
    Computes the share of wrongly predicted labels.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label. If this is not what you want, consider using
    :class:`~pytorch_lightning.metrics.classification.Accuracy`.

    Accepts all input types listed in :ref:`metrics:Input types`.

    Args:
        threshold:
            Threshold probability value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. If ``logits=True``,
            this value is transformed to logits by ``logit_t = ln(t / (1-t))``. Default: 0.5
        logits:
            If predictions are floats, whether they are probabilities or logits. Default ``True``.
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
        Update state with predictions and targets. See :ref:`metrics:Input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, logits, or labels)
            target: Ground truth values
        """
        preds, target, _ = _input_format_classification(preds, target, threshold=self.threshold, logits=self.logits)

        self.correct += (preds == target).sum()
        self.total += preds.numel()

    def compute(self) -> torch.Tensor:
        """
        Computes hamming loss based on inputs passed in to ``update`` previously.
        """
        return 1 - self.correct.float() / self.total


class TopKAccuracy(Metric):
    """
    Computes Top-k classification score for (multi-dimensional) multi-class data.

    Top-k accuracy is the share of correctly predicted samples, where a sample is
    considered correctly predicted if the ground truth label (class) is among the
    ``k``  highest probability labels for the sample.

    Accepts only multi-class and multi-dimensional multi-class inputs with predictions
    as probabilities (as defined in :ref:`metrics:Input types`).

    For multi-dimensional multi-class inputs a ``subset_accuracy`` flag is provided:
    if ``True``, all class predictions across the extra dimension(s) must be correct
    for each sample, for the sample to be counted as correct - this is the same as
    the :class:`~pytorch_lightning.metrics.classification.Accuracy` metric if ``k=1``.

    If ``subset_accuracy=False`` (default), then each sample's accuracy is the share of
    the correct class predictions across the extra dimension(s). This is equivalent to
    computing the metric "globally", with the ``N`` dimension and all extra dimensions
    (``...``) being unrolled into a new sample dimension.

    Args:
        k:
            Number of highest probability predictions considered to find the correct label.
            Default 2
        subset_accuracy:
            Determines how the metric is computed for multi-dimensional multi-class data:

            - If ``False`` [default], then each sample's accuracy is the share of
              the correct class predictions across the extra dimension(s). This is equivalent to
              computing the metric "globally", with the ``N`` dimension and all extra dimensions
              (``...``) being unrolled into a new sample dimension.

            - If ``True``, all class predictions across the extra dimension(s) must be correct
              for each sample, for the sample to be counted as correct - this is the same as
              the :class:`~pytorch_lightning.metrics.classification.Accuracy` metric if ``k=1``.

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
        >>> preds = torch.tensor([[-1, 0, -2], [0.5, -2, 5.0], [-1, 2, 1]]) # preds are logits
        >>> accuracy = TopKAccuracy()
        >>> accuracy(preds, target)
        tensor(0.6667)
    """

    def __init__(
        self,
        k: int = 2,
        subset_accuracy=False,
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
        self.subset_accuracy = subset_accuracy

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Accepts only multi-class and multi-dimensional multi-class inputs (as defined
        in :ref:`metrics:Input types`).

        Args:
            preds: Predictions from model (probabilities, logits, or labels)
            target: Ground truth values
        """
        preds, target, mode = _input_format_classification(preds, target, logits=True, top_k=self.k)

        if "multi-dim" not in mode or not self.subset_accuracy:
            self.correct += (preds * target).sum()
            self.total += int(preds.numel() / preds.shape[1])
        else:
            extra_dims = list(range(1, len(preds.shape)))
            sample_correct = (preds * target).sum(dim=extra_dims)

            self.correct += (sample_correct == preds[0, 0].numel()).sum()
            self.total += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """
        Computes top k accuracy  based on inputs passed in to ``update`` previously.
        """
        return self.correct.float() / self.total
