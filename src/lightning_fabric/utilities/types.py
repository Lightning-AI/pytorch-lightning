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
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import Protocol, runtime_checkable

from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13, _TORCH_GREATER_EQUAL_2_0

_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]]
_PARAMETERS = Iterator[torch.nn.Parameter]


if torch.distributed.is_available():
    from torch.distributed import ProcessGroup, ReduceOp

    RedOpType = ReduceOp.RedOpType if _TORCH_GREATER_EQUAL_1_13 else object
else:
    ProcessGroup = Any  # type: ignore[assignment,misc]
    ReduceOp = object  # type: ignore[assignment,misc] # we are using isinstance check once
    RedOpType = object


_DictKey = TypeVar("_DictKey")


@runtime_checkable
class _Stateful(Protocol[_DictKey]):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    def state_dict(self) -> Dict[_DictKey, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[_DictKey, Any]) -> None:
        ...


@runtime_checkable
class CollectibleGroup(Protocol):
    def size(self) -> int:
        ...

    def rank(self) -> int:
        ...


# Inferred from `torch.optim.lr_scheduler.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class LRScheduler(_Stateful[str], Protocol):
    optimizer: Optimizer
    base_lrs: List[float]

    def __init__(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> None:
        ...

    def step(self, epoch: Optional[int] = None) -> None:
        ...


_TORCH_LRSCHEDULER = (
    torch.optim.lr_scheduler.LRScheduler if _TORCH_GREATER_EQUAL_2_0 else torch.optim.lr_scheduler._LRScheduler
)


# Inferred from `torch.optim.lr_scheduler.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class ReduceLROnPlateau(_Stateful[str], Protocol):
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


@runtime_checkable
class Steppable(Protocol):
    """To structurally type ``optimizer.step()``"""

    # Inferred from `torch.optim.optimizer.pyi`
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        ...


@runtime_checkable
class Optimizable(Steppable, Protocol):
    """To structurally type ``optimizer``"""

    param_groups: List[Dict[Any, Any]]
    defaults: Dict[Any, Any]
    state: Dict[Any, Any]

    def state_dict(self) -> Dict[str, Dict[Any, Any]]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Dict[Any, Any]]) -> None:
        ...
