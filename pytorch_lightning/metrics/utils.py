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
from typing import List, Optional

import torch
from torchmetrics.utilities.data import dim_zero_cat as __dim_zero_cat
from torchmetrics.utilities.data import dim_zero_mean as __dim_zero_mean
from torchmetrics.utilities.data import dim_zero_sum as __dim_zero_sum
from torchmetrics.utilities.data import to_onehot as __to_onehot

from pytorch_lightning.utilities import rank_zero_warn


def dim_zero_cat(x):
    rank_zero_warn(
        "This `dim_zero_cat` was deprecated since v1.3.0 and it will be removed in v1.5.0", DeprecationWarning
    )
    return __dim_zero_cat(x)


def dim_zero_sum(x):
    rank_zero_warn(
        "This `dim_zero_sum` was deprecated since v1.3.0 and it will be removed in v1.5.0", DeprecationWarning
    )
    return __dim_zero_sum(x)


def dim_zero_mean(x):
    rank_zero_warn(
        "This `dim_zero_mean` was deprecated since v1.3.0 and it will be removed in v1.5.0", DeprecationWarning
    )
    return __dim_zero_mean(x)


def get_group_indexes(idx: torch.Tensor) -> List[torch.Tensor]:
    """
    Given an integer `torch.Tensor` `idx`, return a `torch.Tensor` of indexes for
    each different value in `idx`.

    Args:
        idx: a `torch.Tensor` of integers

    Return:
        A list of integer `torch.Tensor`s

    Example:

        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> groups = get_group_indexes(indexes)
        >>> groups
        [tensor([0, 1, 2]), tensor([3, 4, 5, 6])]
    """

    indexes = dict()
    for i, _id in enumerate(idx):
        _id = _id.item()
        if _id in indexes:
            indexes[_id] += [i]
        else:
            indexes[_id] = [i]
    return [torch.tensor(x, dtype=torch.int64) for x in indexes.values()]


def to_onehot(
    label_tensor: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    r"""
    .. warning:: This function is deprecated, use ``torchmetrics.utilities.data.to_onehot``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `to_onehot` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.data.to_onehot`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __to_onehot(label_tensor=label_tensor, num_classes=num_classes)


def select_topk(prob_tensor: torch.Tensor, topk: int = 1, dim: int = 1) -> torch.Tensor:
    """
    Convert a probability tensor to binary by selecting top-k highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type torch.int32

    Example:

        >>> from pytorch_lightning.metrics.utils import select_topk
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)
    """
    zeros = torch.zeros_like(prob_tensor)
    topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()


def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    """
    Converts a tensor of probabilities to a dense label tensor

    Args:
        tensor: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]

    Example:

        >>> from pytorch_lightning.metrics.utils import to_categorical
        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])
    """
    return torch.argmax(tensor, dim=argmax_dim)


def get_num_classes(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
) -> int:
    """
    Calculates the number of classes for a given prediction and target tensor.

    Args:
        pred: predicted values
        target: true labels
        num_classes: number of classes if known

    Return:
        An integer that represents the number of classes.
    """
    num_target_classes = int(target.max().detach().item() + 1)
    num_pred_classes = int(pred.max().detach().item() + 1)
    num_all_classes = max(num_target_classes, num_pred_classes)

    if num_classes is None:
        num_classes = num_all_classes
    elif num_classes != num_all_classes:
        rank_zero_warn(
            f"You have set {num_classes} number of classes which is"
            f" different from predicted ({num_pred_classes}) and"
            f" target ({num_target_classes}) number of classes",
            RuntimeWarning,
        )
    return num_classes


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
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'`` or ``None``: returns calculated metric per class

    Raises:
        ValueError:
            If ``class_reduction`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"`` or ``None``.
    """
    valid_reduction = ("micro", "macro", "weighted", "none", None)
    if class_reduction == "micro":
        fraction = torch.sum(num) / torch.sum(denom)
    else:
        fraction = num / denom

    # We need to take care of instances where the denom can be 0
    # for some (or all) classes which will produce nans
    fraction[fraction != fraction] = 0

    if class_reduction == "micro":
        return fraction
    elif class_reduction == "macro":
        return torch.mean(fraction)
    elif class_reduction == "weighted":
        return torch.sum(fraction * (weights.float() / torch.sum(weights)))
    elif class_reduction == "none" or class_reduction is None:
        return fraction

    raise ValueError(
        f"Reduction parameter {class_reduction} unknown."
        f" Choose between one of these: {valid_reduction}"
    )
