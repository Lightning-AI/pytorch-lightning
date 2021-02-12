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
from typing import Tuple

import torch

from pytorch_lightning.metrics.utils import _input_format_classification_one_hot, class_reduce


def _fbeta_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
    multilabel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, target = _input_format_classification_one_hot(num_classes, preds, target, threshold, multilabel)
    true_positives = torch.sum(preds * target, dim=1)
    predicted_positives = torch.sum(preds, dim=1)
    actual_positives = torch.sum(target, dim=1)
    return true_positives, predicted_positives, actual_positives


def _fbeta_compute(
    true_positives: torch.Tensor,
    predicted_positives: torch.Tensor,
    actual_positives: torch.Tensor,
    beta: float = 1.0,
    average: str = "micro"
) -> torch.Tensor:
    if average == "micro":
        precision = true_positives.sum().float() / predicted_positives.sum()
        recall = true_positives.sum().float() / actual_positives.sum()
    else:
        precision = true_positives.float() / predicted_positives
        recall = true_positives.float() / actual_positives

    num = (1 + beta**2) * precision * recall
    denom = beta**2 * precision + recall
    return class_reduce(num, denom, weights=actual_positives, class_reduction=average)


def fbeta(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    beta: float = 1.0,
    threshold: float = 0.5,
    average: str = "micro",
    multilabel: bool = False
) -> torch.Tensor:
    """
    Computes f_beta metric.

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        preds: predictions from model (probabilities, or labels)
        target: ground truth labels
        num_classes: Number of classes in the dataset.
        beta: Beta coefficient in the F measure.
        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5

        average:
            - ``'micro'`` computes metric globally
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``'none'`` or ``None`` computes and returns the metric per class

        multilabel: If predictions are from multilabel classification.

    Example:

        >>> from pytorch_lightning.metrics.functional import fbeta
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> fbeta(preds, target, num_classes=3, beta=0.5)
        tensor(0.3333)

    """
    true_positives, predicted_positives, actual_positives = _fbeta_update(
        preds, target, num_classes, threshold, multilabel
    )
    return _fbeta_compute(true_positives, predicted_positives, actual_positives, beta, average)


def f1(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
    average: str = "micro",
    multilabel: bool = False
) -> torch.Tensor:
    """
    Computes F1 metric. F1 metrics correspond to a equally weighted average of the
    precision and recall scores.

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        preds: predictions from model (probabilities, or labels)
        target: ground truth labels
        num_classes: Number of classes in the dataset.
        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5

        average:
            - ``'micro'`` computes metric globally
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``'none'`` or ``None`` computes and returns the metric per class

        multilabel: If predictions are from multilabel classification.

    Example:
        >>> from pytorch_lightning.metrics.functional import f1
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1(preds, target, num_classes=3)
        tensor(0.3333)
    """
    return fbeta(preds, target, num_classes, 1.0, threshold, average, multilabel)
