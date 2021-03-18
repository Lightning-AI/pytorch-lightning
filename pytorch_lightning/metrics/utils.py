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
from torchmetrics.utilities.data import dim_zero_cat as _dim_zero_cat
from torchmetrics.utilities.data import dim_zero_mean as _dim_zero_mean
from torchmetrics.utilities.data import dim_zero_sum as _dim_zero_sum
from torchmetrics.utilities.data import get_num_classes as _get_num_classes
from torchmetrics.utilities.data import select_topk as _select_topk
from torchmetrics.utilities.data import to_categorical as _to_categorical
from torchmetrics.utilities.data import to_onehot as _to_onehot
from torchmetrics.utilities.distributed import class_reduce as _class_reduce
from torchmetrics.utilities.distributed import reduce as _reduce

from pytorch_lightning.utilities.deprecation import deprecated


@deprecated(target=_dim_zero_cat, ver_deprecate="1.3.0", ver_remove="1.5.0")
def dim_zero_cat(x):
    pass


@deprecated(target=_dim_zero_sum, ver_deprecate="1.3.0", ver_remove="1.5.0")
def dim_zero_sum(x):
    pass


@deprecated(target=_dim_zero_mean, ver_deprecate="1.3.0", ver_remove="1.5.0")
def dim_zero_mean(x):
    pass


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


@deprecated(target=_to_onehot, ver_deprecate="1.3.0", ver_remove="1.5.0")
def to_onehot(label_tensor: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.to_onehot`. Will be removed in v1.5.0.
    """


@deprecated(target=_select_topk, ver_deprecate="1.3.0", ver_remove="1.5.0")
def select_topk(prob_tensor: torch.Tensor, topk: int = 1, dim: int = 1) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.select_topk`. Will be removed in v1.5.0.
    """


@deprecated(target=_to_categorical, ver_deprecate="1.3.0", ver_remove="1.5.0")
def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.to_categorical`. Will be removed in v1.5.0.
    """


@deprecated(target=_get_num_classes, ver_deprecate="1.3.0", ver_remove="1.5.0")
def get_num_classes(pred: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None) -> int:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.get_num_classes`. Will be removed in v1.5.0.
    """


@deprecated(target=_reduce, ver_deprecate="1.3.0", ver_remove="1.5.0")
def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.reduce`. Will be removed in v1.5.0.
    """


@deprecated(target=_class_reduce, ver_deprecate="1.3.0", ver_remove="1.5.0")
def class_reduce(
    num: torch.Tensor, denom: torch.Tensor, weights: torch.Tensor, class_reduction: str = "none"
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.class_reduce`. Will be removed in v1.5.0.
    """
