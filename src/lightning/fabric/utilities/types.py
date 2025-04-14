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
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor

# TODO: Unused import, but lightning_habana imports these from here
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau  # noqa: F401
from typing_extensions import TypeAlias, overload

UntypedStorage: TypeAlias = torch.UntypedStorage

_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[
    Union[_DEVICE, Callable[[UntypedStorage, str], Optional[UntypedStorage]], dict[_DEVICE, _DEVICE]]
]
_PARAMETERS = Iterator[torch.nn.Parameter]

if torch.distributed.is_available():
    from torch.distributed import ProcessGroup, ReduceOp

    RedOpType: TypeAlias = ReduceOp.RedOpType
else:
    ProcessGroup = Any  # type: ignore[assignment,misc]
    ReduceOp = RedOpType = object  # type: ignore[assignment,misc] # we are using isinstance check once

_DictKey = TypeVar("_DictKey")


@runtime_checkable
class _Stateful(Protocol[_DictKey]):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    def state_dict(self) -> dict[_DictKey, Any]: ...

    def load_state_dict(self, state_dict: dict[_DictKey, Any]) -> None: ...


@runtime_checkable
class CollectibleGroup(Protocol):
    def size(self) -> int: ...

    def rank(self) -> int: ...


@runtime_checkable
class Steppable(Protocol):
    """To structurally type ``optimizer.step()``"""

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]: ...


@runtime_checkable
class Optimizable(Steppable, Protocol):
    """To structurally type ``optimizer``"""

    param_groups: list[dict[Any, Any]]
    defaults: dict[Any, Any]
    state: defaultdict[Tensor, Any]

    def state_dict(self) -> dict[str, dict[Any, Any]]: ...

    def load_state_dict(self, state_dict: dict[str, dict[Any, Any]]) -> None: ...
