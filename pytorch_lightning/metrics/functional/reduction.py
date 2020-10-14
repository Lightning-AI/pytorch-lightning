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
    if reduction == 'elementwise_mean':
        return torch.mean(to_reduce)
    if reduction == 'none':
        return to_reduce
    if reduction == 'sum':
        return torch.sum(to_reduce)
    raise ValueError('Reduction parameter unknown.')


def class_reduce(num: torch.Tensor,
                 denom: torch.Tensor,
                 weights: torch.Tensor,
                 class_reduction: str = 'none') -> torch.Tensor:
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
    valid_reduction = ('micro', 'macro', 'weighted', 'none')
    if class_reduction == 'micro':
        return torch.sum(num) / torch.sum(denom)

    # For the rest we need to take care of instances where the denom can be 0
    # for some classes which will produce nans for that class
    fraction = num / denom
    fraction[fraction != fraction] = 0
    if class_reduction == 'macro':
        return torch.mean(fraction)
    elif class_reduction == 'weighted':
        return torch.sum(fraction * (weights / torch.sum(weights)))
    elif class_reduction == 'none':
        return fraction

    raise ValueError(f'Reduction parameter {class_reduction} unknown.'
                     f' Choose between one of these: {valid_reduction}')
