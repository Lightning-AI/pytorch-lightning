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
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.overrides import TorchFunctionMode
from typing_extensions import override

from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.types import _DEVICE


# From https://lernapparat.de/faster-model-init by Thomas Viehmann
class _EmptyInit(TorchFunctionMode):
    """Initialize `nn.Module` with empty tensors, i.e., uninitialized memory.

    Example::

        with _EmptyInit():
            model = BigModel()
        model.load_state_dict(torch.load("checkpoint.pt"))

    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

    @override
    def __torch_function__(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        kwargs = kwargs or {}
        if not self.enabled:
            return func(*args, **kwargs)
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            return args[0]
        return func(*args, **kwargs)


def _materialize(module: Module, device: _DEVICE) -> None:
    """Materialize a module."""
    module.to_empty(device=device, recurse=False)
    if not hasattr(module, "reset_parameters"):
        raise TypeError(
            f"Materialization requires that the `{type(module).__name__}.reset_parameters` method is implemented."
            " This method is used to initialize any children parameters or buffers in this module."
        )
    module.reset_parameters()


def _materialize_meta_tensors(module: Module, device: _DEVICE) -> None:
    """Materialize all tensors in a given module."""
    for module in module.modules():
        if _has_meta_device_parameters_or_buffers(module, recurse=False):
            _materialize(module, device)


def _materialize_distributed_module(module: Module, device: torch.device) -> None:
    # Reference: https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md#meta-device-initialization
    # TODO: Introduce `Fabric.materialize(module)` to give user control when materialization should happen
    # TODO: Make `torchmetrics.Metric` compatible with the `to_empty()` + `reset_parameters()` semantics
    if not _has_meta_device_parameters_or_buffers(module):
        return

    module.to_empty(device=device)  # has to be called on the root module

    uninitialized_modules = set()
    for submodule in module.modules():
        if all(False for _ in itertools.chain(submodule.parameters(recurse=False), submodule.buffers(recurse=False))):
            # module has no parameters or buffers
            continue
        if callable(reset_method := getattr(submodule, "reset_parameters", None)):
            reset_method()
        else:
            uninitialized_modules.add(type(submodule).__name__)

    if uninitialized_modules:
        rank_zero_warn(
            "Parameter initialization incomplete. The following modules have parameters or buffers with uninitialized"
            " memory because they don't define a `reset_parameters()` method for re-initialization:"
            f" {', '.join(uninitialized_modules)}"
        )


def _has_meta_device_parameters_or_buffers(obj: Union[Module, Optimizer], recurse: bool = True) -> bool:
    if isinstance(obj, Optimizer):
        return any(
            t.is_meta for param_group in obj.param_groups for t in param_group["params"] if isinstance(t, Parameter)
        )
    if isinstance(obj, Module):
        return any(t.is_meta for t in itertools.chain(obj.parameters(recurse=recurse), obj.buffers(recurse=recurse)))
    raise TypeError(f"Expected `torch.nn.Module` or `torch.optim.Optimizer`, got: {type(obj).__name__}")
