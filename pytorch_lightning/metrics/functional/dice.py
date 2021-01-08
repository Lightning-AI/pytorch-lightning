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
from pytorch_lightning.metrics.functional import f1

def dice_coefficient(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
    average: str = "micro",
    multilabel: bool = False
) -> torch.Tensor:
    """
    Computes Dice Coefficient.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
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

    Example:
        >>> from pytorch_lightning.metrics.functional import f1
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1(preds, target, num_classes=3)
        tensor(0.3333)
    """
    return f1(preds, target, num_classes, threshold, average, multilabel)
