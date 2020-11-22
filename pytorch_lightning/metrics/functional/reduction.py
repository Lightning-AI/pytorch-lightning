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
from typing import Optional

import numpy as np
import torch


def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduces a given tensor by a given reduction method

    Args:
        to_reduce : the tensor, which shall be reduced
       reduction :  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given
    """
    if reduction == "elementwise_mean":
        return torch.mean(to_reduce)
    if reduction == "none":
        return to_reduce
    if reduction == "sum":
        return torch.sum(to_reduce)
    raise ValueError("Reduction parameter unknown.")


def class_reduce(
    num: torch.Tensor, denom: torch.Tensor, weights: torch.Tensor, class_reduction: str = "none"
) -> torch.Tensor:
    """
    Function used to reduce classification metrics of the form `num / denom * weights`.
    For example for calculating standard accuracy the num would be number of
    true positives per class, denom would be the support per class, and weights
    would be a tensor of 1s

    Args:
        num: numerator tensor
        decom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'``: returns calculated metric per class

    """
    valid_reduction = ("micro", "macro", "weighted", "none")
    if class_reduction == "micro":
        return torch.sum(num) / torch.sum(denom)

    # For the rest we need to take care of instances where the denom can be 0
    # for some classes which will produce nans for that class
    fraction = num / denom
    fraction[fraction != fraction] = 0
    if class_reduction == "macro":
        return torch.mean(fraction)
    elif class_reduction == "weighted":
        return torch.sum(fraction * (weights / torch.sum(weights)))
    elif class_reduction == "none":
        return fraction

    raise ValueError(
        f"Reduction parameter {class_reduction} unknown." f" Choose between one of these: {valid_reduction}"
    )


def _reduce_scores(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    weights: torch.Tensor,
    average: str,
    mdmc_average: Optional[str],
    zero_division: int,
) -> torch.Tensor:
    """Reduces scores of type numerator/denominator (with possible weighting).

    First, scores are computed by dividing the numerator by denominator. If
    denominator is zero, then the score is set to the value of zero_division
    parameters.

    If average='micro' or 'none', no reduction is needed. In case of 'none',
    scores for classes whose weights are negative are set to nan.

    If average='macro' or 'weighted', the scores across each classes are
    averaged (with weights). The scores for classes whose weights are
    negative are ignored in averaging.

    If average='samples', the scores across all samples are averaged.

    In case if mdmc_average='samplewise', then the transformations mentioned
    above are first applied across dimension 1, and the scores then averaged
    across dimension 0.

    Parameters
    ----------
    numerator
        A tensor with elements that are the upper part of the quotient
    denominator
        A tensor with elements that are the lower part of the quotient
    weights
        A tensor of weights for each class - will be used for weighting
        only if average='weighted'.

        If a class is to be ignored (in case of macro or weighted average),
        that class should have a negative weight. If average=none or None,
        classes with negative weights will get a score of nan
    average
        The method to average the scores. Should be one of 'micro', 'macro',
        'weighted', 'none', None, 'samples'
    mdmc_average
        The method to average the scores if inputs were multi-dimensional multi-class.
        Should be either 'global' or 'samplewise'. If inputs were not
        multi-dimensional multi-class, it should be None
    zero_division
        Should be either zero (if there is zero division set metric to 0), or 1W
    """
    numerator, denominator = numerator.double(), denominator.double()
    weights = weights.double()

    zero_div_mask = denominator == 0
    denominator = torch.where(zero_div_mask, 1.0, denominator)

    scores = numerator / denominator
    scores = torch.where(zero_div_mask, float(zero_division), scores)

    ignore_mask = weights < 0

    weights = torch.where(ignore_mask, 0.0, 1.0 if average == "macro" else weights)
    weights = weights.double()
    weights_sum = weights.sum(dim=-1, keepdims=True)

    # In case if we ignore the only positive class (sum of weights is 0),
    # return zero_division - this is to be consistent with sklearn and
    # pass the tests
    weights_sum = torch.where(weights_sum == 0, 1.0, weights_sum)
    weights = weights / weights_sum

    if average in ["none", None]:
        scores = torch.where(ignore_mask, np.nan, scores)

    elif average in ["macro", "weighted"]:
        scores = (scores * weights).sum(dim=-1)

    elif average == "samples":
        scores = scores.mean()

    if mdmc_average == "samplewise":
        scores = scores.mean()

    return scores