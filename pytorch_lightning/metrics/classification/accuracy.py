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
from pytorch_lightning.metrics.functional.accuracy import _accuracy_update, _accuracy_compute


class Accuracy(Metric):
    r"""
    Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    For multi-class and multi-dimensional multi-class data with probability predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability items are considered to find the correct label.

    This metric generalizes to subset accuracy for multilabel data: for the sample to be counted as
    correct, all labels in that sample have to be correctly predicted. Consider using :class:`~pytorch_lightning.metrics.classification.HammingLoss`
    is this is not what you want. In a multi-dimensional multi-class case, the `mdmc_accuracy` parameters
    gives you a choice between computing the subset accuracy or counting each sample on the extra
    axis separately.

    Accepts all input types listed in :ref:`metrics:Input types`.

    Args:
        threshold:
            Threshold probability value for transforming probability predictions to binary
            `(0,1)` predictions, in the case of binary or multi-label inputs. Default: `0.5`
        top_k:
            Number of highest probability predictions considered to find the correct label, relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        mdmc_accuracy:
            Determines how should the extra dimension be handled in case of multi-dimensional multi-class
            inputs. Options are ``"global"`` or ``"subset"``.

            If ``"global"``, then the inputs are treated as if the sample (``N``) and the extra dimension
            were unrolled into a new sample dimension.

            If ``"subset"``, then the equivalent of subset accuracy is performed for each sample on the
            ``N`` dimension - that is, for the sample to count as correct, all labels on its extra dimension
            must be predicted correctly (the ``top_k`` option still applies here). The final score is then
            simply the number of totally correctly predicted samples.
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
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        mdmc_accuracy: str = "global",
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

        if not 0 <= threshold <= 1:
            raise ValueError("The `threshold` should lie in the [0,1] interval.")

        self.threshold = threshold
        self.top_k = top_k

        if mdmc_accuracy not in ["global", "subset"]:
            raise ValueError("The `mdmc_accuracy` should be either 'subset' or 'global'.")

        self.mdmc_accuracy = mdmc_accuracy

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`metrics:Input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth values
        """

        correct, total = _accuracy_update(
            preds, target, threshold=self.threshold, top_k=self.top_k, mdmc_accuracy=self.mdmc_accuracy
        )

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        return _accuracy_compute(self.correct, self.total)
