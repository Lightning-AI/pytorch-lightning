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
from typing import Tuple, Optional

import torch
from pytorch_lightning.metrics.classification.helpers import _input_format_classification


def _accuracy_update(
    preds: torch.Tensor, target: torch.Tensor, threshold: float, top_k: Optional[int], mdmc_accuracy: str
) -> Tuple[torch.Tensor, torch.Tensor]:

    preds, target, mode = _input_format_classification(preds, target, threshold=threshold, top_k=top_k)

    if mode in ["binary", "multi-label"]:
        correct = (preds == target).all(dim=1).sum()
        total = target.shape[0]
    elif mdmc_accuracy == "global":
        correct = (preds * target).sum()
        total = target.sum()
    elif mdmc_accuracy == "subset":
        extra_dims = list(range(1, len(preds.shape)))
        sample_correct = (preds * target).sum(dim=extra_dims)
        sample_total = target.sum(dim=extra_dims)

        correct = (sample_correct == sample_total).sum()
        total = target.shape[0]

    return (torch.tensor(correct, device=preds.device), torch.tensor(total, device=preds.device))


def _accuracy_compute(correct: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    return correct.float() / total


def accuracy(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    mdmc_accuracy: str = "subset",
) -> torch.Tensor:
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
    is this is not what you want. In multi-dimensional multi-class case the `mdmc_accuracy` parameters
    gives you a choice between computing the subset accuracy, or counting each sample on the extra
    axis separately.

    Accepts all input types listed in :ref:`metrics:Input types`.

    Args:
        preds: Predictions from model (probabilities, or labels)
        target: Ground truth values
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
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

    Example:

        >>> from pytorch_lightning.metrics.functional import accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy(preds, target)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy(preds, target, top_k=2)
        tensor(0.6667)
    """

    if mdmc_accuracy not in ["global", "subset"]:
        raise ValueError("The `mdmc_accuracy` should be either 'subset' or 'global'.")

    correct, total = _accuracy_update(preds, target, threshold, top_k, mdmc_accuracy)
    return _accuracy_compute(correct, total)
