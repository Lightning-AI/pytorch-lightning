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
from typing import Tuple, Union

import torch
from pytorch_lightning.metrics.classification.utils import _input_format_classification

################################
# Accuracy
################################


def _accuracy_update(
    preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, logits: bool = True
) -> Tuple[torch.Tensor, int]:

    preds, target, _ = _input_format_classification(preds, target, threshold=threshold, logits=logits)

    extra_dims = list(range(1, len(preds.shape)))
    sample_correct = (preds == target).sum(dim=extra_dims)

    correct = (sample_correct == preds[0].numel()).sum()
    total = preds.shape[0]

    return (correct, total)


def _accuracy_compute(correct: torch.Tensor, total: Union[int, torch.Tensor]) -> torch.Tensor:
    return correct.float() / total


def accuracy(preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, logits: bool = True) -> torch.Tensor:
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
            If predictions are floats, whether they are probabilities or logits. Default ``True``
            (predictions are logits).

    Example:

        >>> from pytorch_lightning.metrics.functional import accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy(preds, target)
        tensor(0.5000)

    """

    correct, total = _accuracy_update(preds, target, threshold, logits)
    return _accuracy_compute(correct, total)


################################
# Hamming loss
################################


def _hamming_loss_update(
    preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, logits: bool = True
) -> Tuple[torch.Tensor, int]:
    preds, target, _ = _input_format_classification(preds, target, threshold=threshold, logits=logits)

    correct = (preds == target).sum()
    total = preds.numel()

    return correct, total


def _hamming_loss_compute(correct: torch.Tensor, total: Union[int, torch.Tensor]) -> torch.Tensor:
    return 1 - correct.float() / total


def hamming_loss(
    preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, logits: bool = True
) -> torch.Tensor:
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
            If predictions are floats, whether they are probabilities or logits. Default ``True``
            (predictions are logits).

    Example:

        >>> from pytorch_lightning.metrics.functional import hamming_loss
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> hamming_loss(preds, target)
        tensor(0.2500)

    """

    correct, total = _hamming_loss_update(preds, target, threshold, logits)
    return _hamming_loss_compute(correct, total)


################################
# TopK Accuracy
################################


def _topk_accuracy_update(
    preds: torch.Tensor, target: torch.Tensor, subset_accuracy: bool = False, k: int = 2
) -> Tuple[torch.Tensor, int]:
    preds, target, mode = _input_format_classification(preds, target, logits=True, top_k=k)

    if "multi-dim" not in mode or not subset_accuracy:
        correct = (preds * target).sum()
        total = int(preds.numel() / preds.shape[1])
    else:
        extra_dims = list(range(1, len(preds.shape)))
        sample_correct = (preds * target).sum(dim=extra_dims)

        correct = (sample_correct == preds[0, 0].numel()).sum()
        total = preds.shape[0]

    return correct, total


def _topk_accuracy_compute(correct: torch.Tensor, total: Union[int, torch.Tensor]) -> torch.Tensor:
    return correct.float() / total


def topk_accuracy(preds: torch.Tensor, target: torch.Tensor, subset_accuracy: bool = False, k: int = 2) -> torch.Tensor:
    """
    Computes Top-k classification score for (multi-dimensional) multi-class data.

    Top-k accuracy is the share of correctly predicted samples, where a sample is
    considered correctly predicted if the ground truth label (class) is among the
    ``k``  highest probability labels for the sample.

    Accepts only multi-class and multi-dimensional multi-class inputs with predictions
    as probabilities (as defined in :ref:`metrics:Input types`).

    For multi-dimensional multi-class inputs a ``subset_accuracy`` flag is provided,
    which determines how the reduction is applied.

    Args:
        k:
            Number of highest probability predictions considered to find the correct label.
            Default 2
        subset_accuracy:
            Determines how the metric is computed for multi-dimensional multi-class data:

            - If ``False`` [default], then the the ``N`` dimension and all extra dimensions
              (``...``) being unrolled into a new sample dimension, and the metric is computed
              on these new unrolled inputs (aking to the ``global`` setting in some other
              metrics, such as :class:`~pytorch_lightning.metrics.classification.Recall`).

            - If ``True``, all class predictions across the extra dimension(s) must be correct
              for each sample, for the sample to be counted as correct - this is the same as
              the :class:`~pytorch_lightning.metrics.classification.Accuracy` metric in the case
              when ``k=1``.

    Example:

        >>> from pytorch_lightning.metrics.functional import topk_accuracy
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1, 0, -2], [0.5, -2, 5.0], [-1, 2, 1]]) # preds are logits
        >>> topk_accuracy(preds, target)
        tensor(0.6667)
    """

    correct, total = _topk_accuracy_update(preds, target, subset_accuracy, k)
    return _topk_accuracy_compute(correct, total)
