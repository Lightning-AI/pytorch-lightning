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
from functools import partial
from typing import Optional

import torch
from deprecate import deprecated
from torchmetrics.utilities.data import dim_zero_cat as _dim_zero_cat
from torchmetrics.utilities.data import dim_zero_mean as _dim_zero_mean
from torchmetrics.utilities.data import dim_zero_sum as _dim_zero_sum
from torchmetrics.utilities.data import get_num_classes as _get_num_classes
from torchmetrics.utilities.data import select_topk as _select_topk
from torchmetrics.utilities.data import to_categorical as _to_categorical
from torchmetrics.utilities.data import to_onehot as _to_onehot
from torchmetrics.utilities.distributed import class_reduce as _class_reduce
from torchmetrics.utilities.distributed import reduce as _reduce

from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_3, _TORCHMETRICS_LOWER_THAN_0_3

deprecated_metrics = partial(deprecated, deprecated_in="1.3.0", remove_in="1.5.0", stream=rank_zero_deprecation)


@deprecated_metrics(target=_dim_zero_cat)
def dim_zero_cat(x):
    pass


@deprecated_metrics(target=_dim_zero_sum)
def dim_zero_sum(x):
    pass


@deprecated_metrics(target=_dim_zero_mean)
def dim_zero_mean(x):
    pass


@deprecated_metrics(target=_to_onehot)
def to_onehot(label_tensor: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.to_onehot`. Will be removed in v1.5.0.
    """


@deprecated_metrics(target=_select_topk)
def select_topk(prob_tensor: torch.Tensor, topk: int = 1, dim: int = 1) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.select_topk`. Will be removed in v1.5.0.
    """


@deprecated_metrics(target=_to_categorical)
def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.to_categorical`. Will be removed in v1.5.0.
    """


@deprecated_metrics(target=_get_num_classes, skip_if=_TORCHMETRICS_GREATER_EQUAL_0_3)
@deprecated_metrics(target=_get_num_classes, args_mapping=dict(pred="preds"), skip_if=_TORCHMETRICS_LOWER_THAN_0_3)
def get_num_classes(pred: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None) -> int:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.data.get_num_classes`. Will be removed in v1.5.0.
    """


@deprecated_metrics(target=_reduce)
def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.reduce`. Will be removed in v1.5.0.
    """


@deprecated_metrics(target=_class_reduce)
def class_reduce(
    num: torch.Tensor, denom: torch.Tensor, weights: torch.Tensor, class_reduction: str = "none"
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.utilities.class_reduce`. Will be removed in v1.5.0.
    """
