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
from argparse import _ArgumentGroup, ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Type, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing_extensions import Protocol, runtime_checkable

from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER, LRScheduler, ProcessGroup, ReduceLROnPlateau

_NUMBER = Union[int, float]
_METRIC = Union[Metric, Tensor, _NUMBER]
_METRIC_COLLECTION = Union[_METRIC, Mapping[str, _METRIC]]
STEP_OUTPUT = Union[Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]
_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader
_PREDICT_OUTPUT = Union[List[Any], List[List[Any]]]
TRAIN_DATALOADERS = Union[
    DataLoader,
    Sequence[DataLoader],
    Sequence[Sequence[DataLoader]],
    Sequence[Dict[str, DataLoader]],
    Dict[str, DataLoader],
    Dict[str, Dict[str, DataLoader]],
    Dict[str, Sequence[DataLoader]],
]
EVAL_DATALOADERS = Union[DataLoader, Sequence[DataLoader]]
_ADD_ARGPARSE_RETURN = Union[_ArgumentGroup, ArgumentParser]


@runtime_checkable
class TrainingStep(Protocol):
    """This class is used to detect if an object implements the `training_step` hook using `isinstance(model,
    TrainingStep)`."""

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ...


@runtime_checkable
class ValidationStep(Protocol):
    """This class is used to detect if an object implements the `validation_step` hook using `isinstance(model,
    ValidationStep)`."""

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        ...


@runtime_checkable
class TestStep(Protocol):
    """This class is used to detect if an object implements the `test_step` hook using `isinstance(model,
    TestStep)`."""

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        ...


@runtime_checkable
class PredictStep(Protocol):
    """This class is used to detect if an object implements the `predict_step` hook using `isinstance(model,
    PredictStep)`."""

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ...


# Inferred from `torch.nn.parallel.distributed.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class DistributedDataParallel(Protocol):
    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Optional[List[Union[int, torch.device]]] = None,
        output_device: Optional[Union[int, torch.device]] = None,
        dim: int = 0,
        broadcast_buffers: bool = True,
        process_group: Optional[ProcessGroup] = None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
    ) -> None:
        ...

    @contextmanager
    def no_sync(self) -> Generator:
        ...


# todo: improve LRSchedulerType naming/typing
LRSchedulerTypeTuple = (_TORCH_LRSCHEDULER, torch.optim.lr_scheduler.ReduceLROnPlateau)
LRSchedulerTypeUnion = Union[_TORCH_LRSCHEDULER, torch.optim.lr_scheduler.ReduceLROnPlateau]
LRSchedulerType = Union[Type[_TORCH_LRSCHEDULER], Type[torch.optim.lr_scheduler.ReduceLROnPlateau]]
LRSchedulerPLType = Union[LRScheduler, ReduceLROnPlateau]


@dataclass
class LRSchedulerConfig:
    scheduler: Union[_TORCH_LRSCHEDULER, ReduceLROnPlateau]
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
    # opt_idx assigned internally if not assigned by user
    opt_idx: Optional[int] = None
