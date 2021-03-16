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
from typing import Optional, Tuple

import numpy as np
import torch
from torchmetrics.classification.checks import _check_classification_inputs
from torchmetrics.utilities.data import select_topk, to_onehot

from pytorch_lightning.utilities import LightningEnum


class DataType(LightningEnum):
    """
    Enum to represent data type
    """

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(LightningEnum):
    """
    Enum to represent average method
    """

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"
    SAMPLES = "samples"


class MDMCAverageMethod(LightningEnum):
    """
    Enum to represent multi-dim multi-class average method
    """

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"


def _reduce_stat_scores(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    weights: Optional[torch.Tensor],
    average: str,
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> torch.Tensor:
    """
    Reduces scores of type ``numerator/denominator`` or
    ``weights * (numerator/denominator)``, if ``average='weighted'``.

    Args:
        numerator: A tensor with numerator numbers.
        denominator: A tensor with denominator numbers. If a denominator is
            negative, the class will be ignored (if averaging), or its score
            will be returned as ``nan`` (if ``average=None``).
            If the denominator is zero, then ``zero_division`` score will be
            used for those elements.
        weights:
            A tensor of weights to be used if ``average='weighted'``.
        average:
            The method to average the scores. Should be one of ``'micro'``, ``'macro'``,
            ``'weighted'``, ``'none'``, ``None`` or ``'samples'``. The behavior
            corresponds to `sklearn averaging methods <https://scikit-learn.org/stable/modules/\
model_evaluation.html#multiclass-and-multilabel-classification>`__.
        mdmc_average:
            The method to average the scores if inputs were multi-dimensional multi-class (MDMC).
            Should be either ``'global'`` or ``'samplewise'``. If inputs were not
            multi-dimensional multi-class, it should be ``None`` (default).
        zero_division:
            The value to use for the score if denominator equals zero.
    """
    numerator, denominator = numerator.float(), denominator.float()
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0

    if weights is None:
        weights = torch.ones_like(denominator)
    else:
        weights = weights.float()

    numerator = torch.where(zero_div_mask, torch.tensor(float(zero_division), device=numerator.device), numerator)
    denominator = torch.where(zero_div_mask | ignore_mask, torch.tensor(1.0, device=denominator.device), denominator)
    weights = torch.where(ignore_mask, torch.tensor(0.0, device=weights.device), weights)

    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights = weights / weights.sum(dim=-1, keepdim=True)

    scores = weights * (numerator / denominator)

    # This is in case where sum(weights) = 0, which happens if we ignore the only present class with average='weighted'
    scores = torch.where(torch.isnan(scores), torch.tensor(float(zero_division), device=scores.device), scores)

    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(dim=0)
        ignore_mask = ignore_mask.sum(dim=0).bool()

    if average in (AverageMethod.NONE, None):
        scores = torch.where(ignore_mask, torch.tensor(np.nan, device=scores.device), scores)
    else:
        scores = scores.sum()

    return scores
