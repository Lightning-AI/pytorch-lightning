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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torchmetrics import Metric as _Metric
from torchmetrics import MetricCollection as _MetricCollection

from pytorch_lightning.utilities.distributed import rank_zero_warn


class Metric(_Metric):
    r"""
    This implementation refers to :class:`~torchmetrics.Metric`.

    .. warning:: This metric is deprecated, use ``torchmetrics.Metric``. Will be removed in v1.5.0.
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        rank_zero_warn(
            "This `Metric` was deprecated since v1.3.0 in favor of `torchmetrics.Metric`."
            " It will be removed in v1.5.0", DeprecationWarning
        )
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def __hash__(self):
        return super().__hash__()

    def __add__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __eq__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.gt, self, other)

    def __le__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.le, self, other)

    def __lt__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.mul, self, other)

    def __ne__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric

        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self):
        return self.__inv__()

    def __neg__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(_neg, self, None)

    def __pos__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.abs, self, None)


def _neg(tensor: torch.Tensor):
    return -torch.abs(tensor)


class MetricCollection(_MetricCollection):
    r"""
    This implementation refers to :class:`~torchmetrics.MetricCollection`.

    .. warning:: This metric is deprecated, use ``torchmetrics.MetricCollection``. Will be removed in v1.5.0.
    """

    def __init__(self, metrics: Union[List[Metric], Tuple[Metric], Dict[str, Metric]]):
        rank_zero_warn(
            "This `MetricCollection` was deprecated since v1.3.0 in favor of `torchmetrics.MetricCollection`."
            " It will be removed in v1.5.0", DeprecationWarning
        )
        super().__init__(metrics=metrics)
