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
            if args:
                return args[0]
        return func(*args, **kwargs)


def _validate_reset_parameters(module: Module) -> None:
    if not callable(getattr(module, "reset_parameters", None)):
        raise TypeError(
            f"Materialization requires that the `{type(module).__name__}.reset_parameters` method is implemented."
            " This method is used to initialize any children parameters or buffers in this module."
        )


def _materialize(module: Module, device: _DEVICE) -> None:
    """Materialize a module."""
    _validate_reset_parameters(module)
    module.to_empty(device=device, recurse=False)
    if callable(module.reset_parameters):
        module.reset_parameters()


def _materialize_meta_tensors(module: Module, device: _DEVICE) -> None:
    """Materialize all tensors in a given module."""
    meta_modules: list[Module] = []
    seen_meta_tensors: set[int] = set()
    for submodule in module.modules():
        unseen_meta = [
            id(t)
            for t in itertools.chain(submodule.parameters(recurse=False), submodule.buffers(recurse=False))
            if t.is_meta and id(t) not in seen_meta_tensors
        ]
        if unseen_meta:
            seen_meta_tensors.update(unseen_meta)
            meta_modules.append(submodule)
    for submodule in meta_modules:
        _validate_reset_parameters(submodule)

    materialized_params: dict[int, Parameter] = {}
    materialized_buffers: dict[int, torch.Tensor] = {}
    for submodule in module.modules():
        for name, param in list(submodule._parameters.items()):
            if param is not None and id(param) in materialized_params:
                submodule._parameters[name] = materialized_params[id(param)]
        for name, buf in list(submodule._buffers.items()):
            if buf is not None and id(buf) in materialized_buffers:
                submodule._buffers[name] = materialized_buffers[id(buf)]
        if _has_meta_device_parameters_or_buffers(submodule, recurse=False):
            old_param_ids = {
                name: id(param)
                for name, param in submodule._parameters.items()
                if param is not None and param.is_meta
            }
            old_buf_ids = {
                name: id(buf) for name, buf in submodule._buffers.items() if buf is not None and buf.is_meta
            }
            _materialize(submodule, device)
            for name, old_id in old_param_ids.items():
                new_param = submodule._parameters.get(name)
                if new_param is not None:
                    materialized_params[old_id] = new_param
            for name, old_id in old_buf_ids.items():
                new_buf = submodule._buffers.get(name)
                if new_buf is not None:
                    materialized_buffers[old_id] = new_buf


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
