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
"""
Convention:
 - Do not include any `_TYPE` suffix
 - Types used in public hooks (as those in the `LightningModule` and `Callback`) should be public (no leading `_`)
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Type, Union

import torch
from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup
from torch.optim import Optimizer
from typing_extensions import Protocol, runtime_checkable

_NUMBER = Union[int, float]
_PARAMETERS = Iterator[torch.nn.Parameter]
_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]]


@runtime_checkable
class _Stateful(Protocol):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


# Inferred from `torch.optim.lr_scheduler.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class _LRScheduler(_Stateful, Protocol):
    optimizer: Optimizer
    base_lrs: List[float]

    def __init__(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> None:
        ...

    def step(self, epoch: Optional[int] = None) -> None:
        ...


# Inferred from `torch.optim.lr_scheduler.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class ReduceLROnPlateau(_Stateful, Protocol):
    in_cooldown: bool
    optimizer: Optimizer

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = ...,
        factor: float = ...,
        patience: int = ...,
        verbose: bool = ...,
        threshold: float = ...,
        threshold_mode: str = ...,
        cooldown: int = ...,
        min_lr: float = ...,
        eps: float = ...,
    ) -> None:
        ...

    def step(self, metrics: Union[float, int, Tensor], epoch: Optional[int] = None) -> None:
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
LRSchedulerTypeTuple = (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
LRSchedulerTypeUnion = Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]
LRSchedulerType = Union[Type[torch.optim.lr_scheduler._LRScheduler], Type[torch.optim.lr_scheduler.ReduceLROnPlateau]]
LRSchedulerPLType = Union[_LRScheduler, ReduceLROnPlateau]
