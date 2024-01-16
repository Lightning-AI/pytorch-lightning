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
import itertools
from typing import Any, Callable, Dict, Optional, Sequence
from typing_extensions import override
import torch

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13, _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.utilities.types import _DEVICE

if _TORCH_GREATER_EQUAL_1_13:
    from torch.overrides import TorchFunctionMode
else:
    TorchFunctionMode = object  # type: ignore[misc,assignment]


# From https://lernapparat.de/faster-model-init by Thomas Viehmann
class _EmptyInit(TorchFunctionMode):
    """Initialize `nn.Module` with empty tensors, i.e., uninitialized memory.

    Example::

        with _EmptyInit():
            model = BigModel()
        model.load_state_dict(torch.load("checkpoint.pt"))

    """

    def __init__(self, enabled: bool = True) -> None:
        if not _TORCH_GREATER_EQUAL_1_13:
            raise NotImplementedError("Emtpy weight initialization requires PyTorch >= 1.13.")
        super().__init__()
        self.enabled = enabled

    @override
    def __torch_function__(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ) -> Any:
        kwargs = kwargs or {}
        if not self.enabled:
            return func(*args, **kwargs)
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            return args[0]
        return func(*args, **kwargs)


def _materialize(module: torch.nn.Module, device: _DEVICE) -> None:
    """Materialize a module."""
    if not _TORCH_GREATER_EQUAL_2_1:
        raise RuntimeError("recurse=False requires torch 2.1")
    module.to_empty(device=device, recurse=False)  # type: ignore[arg-type]
    if not hasattr(module, "reset_parameters"):
        raise TypeError(
            f"Materialization requires that the `{type(module).__name__}.reset_parameters` method is implemented."
            " This method is used to initialize any children parameters or buffers in this module."
        )
    module.reset_parameters()


def _materialize_meta_tensors(module: torch.nn.Module, device: _DEVICE) -> None:
    """Materialize all tensors in a given module."""
    for module in module.modules():
        if any(t.is_meta for t in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))):
            _materialize(module, device)
