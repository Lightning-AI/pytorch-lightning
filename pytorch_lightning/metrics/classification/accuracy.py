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
from pytorch_lightning.metrics.functional.accuracy import (
    _accuracy_update,
    _hamming_loss_update,
    _accuracy_compute,
    _hamming_loss_compute,
)


class Accuracy(Metric):
    """
    Computes the share of entirely correctly predicted samples.

    This metric generalizes to subset accuracy for multilabel data, and similarly for
    multi-dimensional multi-class data: for the sample to be counted as correct, the the
    class has to be correctly predicted across all extra dimension for each sample in the
    ``N`` dimension. Consider using :class:`~pytorch_lightning.metrics.classification.HammingLoss`
    is this is not what you want.

    For multi-class and multi-dimensional multi-class data with probability predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability items are considered to find the correct label.

    Accepts all input types listed in :ref:`metrics:Input types`.

    Args:
        top_k:
            Number of highest probability predictions considered to find the correct label, for
            (multi-dimensional) multi-class inputs with probability predictions. Default 1

            If your inputs are not (multi-dimensional) multi-class inputs with probability predictions,
            an error will be raised if ``top_k`` is set to a value other than 1.
        mdmc_accuracy:
            Determines how should the extra dimension be handeled in case of multi-dimensional multi-class
            inputs. Options are ``"global"`` or ``"subset"``.

            If ``"global"``, then the inputs are treated as if the sample (``N``) and the extra dimension
            were unrolled into a new sample dimension.

            If ``"subset"``, than the equivalent of subset accuracy is performed for each sample on the
            ``N`` dimension - that is, for the sample to count as correct, all labels on its extra dimension
            must be predicted correctly (the ``top_k`` option still applies here).
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
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

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy = Accuracy(top_k=2)
        >>> accuracy(preds, target)
        tensor(0.6667)

    """

    def __init__(
        self,
        top_k: int = 1,
        mdmc_accuracy: str = "subset",
        threshold: float = 0.5,
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
        self.top_k = top_k
        self.mdmc_accuracy = mdmc_accuracy

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`metrics:Input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth values
        """

        correct, total = _accuracy_update(preds, target, self.threshold, self.top_k, self.mdmc_accuracy)

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        return _accuracy_compute(self.correct, self.total)


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
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
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

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`metrics:Input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth values
        """
        correct, total = _hamming_loss_update(preds, target, self.threshold)

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        """
        Computes hamming loss based on inputs passed in to ``update`` previously.
        """
        return _hamming_loss_compute(self.correct, self.total)
