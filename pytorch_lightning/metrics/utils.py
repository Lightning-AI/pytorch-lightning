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
from torchmetrics.utilities.data import get_num_classes as __get_num_classes
from torchmetrics.utilities.data import select_topk as __select_topk
from torchmetrics.utilities.data import to_categorical as __to_categorical
from torchmetrics.utilities.data import to_onehot as __to_onehot
from torchmetrics.utilities.distributed import class_reduce as __class_reduce
from torchmetrics.utilities.distributed import reduce as __reduce

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


def to_onehot(label_tensor: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    r"""
    .. warning:: This function is deprecated, use ``torchmetrics.utilities.data.to_onehot``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `to_onehot` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.data.to_onehot`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __to_onehot(label_tensor=label_tensor, num_classes=num_classes)


def select_topk(prob_tensor: torch.Tensor, topk: int = 1, dim: int = 1) -> torch.Tensor:
    r"""
    .. warning::

        This function is deprecated, use ``torchmetrics.utilities.data.select_topk``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `select_topk` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.data.select_topk`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __select_topk(prob_tensor=prob_tensor, topk=topk, dim=dim)


def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    r"""
    .. warning::

        This function is deprecated, use ``torchmetrics.utilities.data.to_categorical``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `to_categorical` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.data.to_categorical`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __to_categorical(tensor=tensor, argmax_dim=argmax_dim)


def get_num_classes(pred: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None) -> int:
    r"""
    .. warning::

        This function is deprecated, use ``torchmetrics.utilities.data.get_num_classes``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `get_num_classes` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.data.get_num_classes`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __get_num_classes(pred=pred, target=target, num_classes=num_classes)


def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    r"""
    .. warning::

        This function is deprecated, use ``torchmetrics.utilities.reduce``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `reduce` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.reduce`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __reduce(to_reduce=to_reduce, reduction=reduction)


def class_reduce(
    num: torch.Tensor, denom: torch.Tensor, weights: torch.Tensor, class_reduction: str = "none"
) -> torch.Tensor:
    r"""
    .. warning::

        This function is deprecated, use ``torchmetrics.utilities.class_reduce``. Will be removed in v1.5.0.
    """
    rank_zero_warn(
        "This `class_reduce` was deprecated since v1.3.0 in favor of `torchmetrics.utilities.class_reduce`."
        " It will be removed in v1.5.0", DeprecationWarning
    )
    return __class_reduce(num=num, denom=denom, weights=weights, class_reduction=class_reduction)
