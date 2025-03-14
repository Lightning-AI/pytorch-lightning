# Copyright The Lightning AI team.
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
"""
Convention:
 - Do not include any `_TYPE` suffix
 - Types used in public hooks (as those in the `LightningModule` and `Callback`) should be public (no leading `_`)
"""

from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torchmetrics import Metric
from typing_extensions import NotRequired, Required

from lightning.fabric.utilities.types import ProcessGroup

_NUMBER = Union[int, float]
_METRIC = Union[Metric, Tensor, _NUMBER]
STEP_OUTPUT = Optional[Union[Tensor, Mapping[str, Any]]]
_EVALUATE_OUTPUT = list[Mapping[str, float]]  # 1 dict per DataLoader
_PREDICT_OUTPUT = Union[list[Any], list[list[Any]]]
TRAIN_DATALOADERS = Any  # any iterable or collection of iterables
EVAL_DATALOADERS = Any  # any iterable or collection of iterables


# Inferred from `torch.nn.parallel.distributed.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class DistributedDataParallel(Protocol):
    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Optional[list[Union[int, torch.device]]] = None,
        output_device: Optional[Union[int, torch.device]] = None,
        dim: int = 0,
        broadcast_buffers: bool = True,
        process_group: Optional[ProcessGroup] = None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
    ) -> None: ...

    @contextmanager
    def no_sync(self) -> Generator: ...


# todo: improve LRSchedulerType naming/typing
LRSchedulerTypeTuple = (LRScheduler, ReduceLROnPlateau)
LRSchedulerTypeUnion = Union[LRScheduler, ReduceLROnPlateau]
LRSchedulerType = Union[type[LRScheduler], type[ReduceLROnPlateau]]
LRSchedulerPLType = Union[LRScheduler, ReduceLROnPlateau]


@dataclass
class LRSchedulerConfig:
    scheduler: Union[LRScheduler, ReduceLROnPlateau]
    # no custom name
    name: Optional[str] = None
    # after epoch is over
    interval: str = "epoch"
    # every epoch/batch
    frequency: int = 1
    # most often not ReduceLROnPlateau scheduler
    reduce_on_plateau: bool = False
    # value to monitor for ReduceLROnPlateau
    monitor: Optional[str] = None
    # enforce that the monitor exists for ReduceLROnPlateau
    strict: bool = True


class LRSchedulerConfigType(TypedDict, total=False):
    scheduler: Required[LRSchedulerTypeUnion]
    name: Optional[str]
    interval: str
    frequency: int
    reduce_on_plateau: bool
    monitor: Optional[str]
    strict: bool


class OptimizerConfig(TypedDict):
    optimizer: Optimizer


class OptimizerLRSchedulerConfig(TypedDict):
    optimizer: Optimizer
    lr_scheduler: Union[LRSchedulerTypeUnion, LRSchedulerConfigType]
    monitor: NotRequired[str]


OptimizerLRScheduler = Optional[
    Union[
        Optimizer,
        Sequence[Optimizer],
        tuple[Sequence[Optimizer], Sequence[Union[LRSchedulerTypeUnion, LRSchedulerConfig]]],
        OptimizerConfig,
        OptimizerLRSchedulerConfig,
        Sequence[OptimizerConfig],
        Sequence[OptimizerLRSchedulerConfig],
    ]
]


class _SizedIterable(Protocol):
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator:
        pass
