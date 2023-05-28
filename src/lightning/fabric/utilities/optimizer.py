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

import contextlib
import functools
import threading
from typing import Iterable, TYPE_CHECKING

from lightning_utilities.core.apply_func import apply_to_collection
import torch
from torch.optim import Optimizer

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.fabric.utilities.types import _DEVICE

if TYPE_CHECKING:
    from lightning.fabric.wrappers import _FabricModule

SUPPORTS_OPTIMIZER_IN_FSDP_BACKWARD = False
try:
    from torch.distributed.fsdp.flat_param import FlatParamHandle
    from torch.distributed.fsdp._traversal_utils import _get_fsdp_handles
    from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
    SUPPORTS_OPTIMIZER_IN_FSDP_BACKWARD = _TORCH_GREATER_EQUAL_2_0
except ImportError:
    pass


def _optimizers_to_device(optimizers: Iterable[Optimizer], device: _DEVICE) -> None:
    """Moves optimizer states for a sequence of optimizers to the device."""
    for opt in optimizers:
        _optimizer_to_device(opt, device)


def _optimizer_to_device(optimizer: Optimizer, device: _DEVICE) -> None:
    """Moves the state of a single optimizer to the device."""
    for p, v in optimizer.state.items():
        optimizer.state[p] = apply_to_collection(v, torch.Tensor, move_data_to_device, device, allow_frozen=True)


def _no_op():
    pass


@contextlib.contextmanager
def _apply_optimizers_during_fsdp_backward(
    optimizers: Iterable[Optimizer],
    module_wrapper: "_FabricModule",
):
    """Call `Optimizer.step` as gradients become available.

    NOTE: This is an EXPERIMENTAL utility and exploits behavior which is not
          part of the FSDP public API. Use at your own risk.

    By moving optimizer step invocation into the backward call we can free
    gradients earlier and reduce peak memory.
    """
    assert SUPPORTS_OPTIMIZER_IN_FSDP_BACKWARD
    apply_lock = threading.Lock()

    from lightning.fabric.wrappers import _FabricModule
    assert isinstance(module_wrapper, _FabricModule)
    param_handles = _get_fsdp_handles(module := module_wrapper._forward_module)
    assert param_handles, f"Module {module} does not appear to contain any FSDP modules."
    fsdp_stream = _get_module_fsdp_state(module)._streams["post_backward"]

    if isinstance(optimizers, Optimizer):
        optimizers = [optimizers]

    # We cannot trigger the optimizer step until all parameters are ready.
    remaining = {}
    for optimizer in optimizers:
        unfinished = {}  # Use Dict as an ordered set.
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in unfinished:
                    assert p not in remaining, f"{p=} is shared between two optimizers."
                    unfinished[p] = None
                    remaining[p] = (optimizer, unfinished)

    def maybe_step(parameters, post_step=lambda: None) -> None:
        for p in tuple(parameters):
            optimizer, unfinished = remaining.pop(p)
            unfinished.pop(p)
            if not unfinished:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                post_step()

    try:
        hook_handles = []
        for h in param_handles:
            assert isinstance(h, FlatParamHandle)
            flat_param = h.flat_param
            assert hasattr(flat_param, "_post_backward_hook_state")
            fsdp_acc_grad, _ = flat_param._post_backward_hook_state

            # We must take `h` and `flat_param` as arguments because Python
            # late binds closures.
            def _opt_hook(h, flat_param, *_unused):
                assert flat_param._post_backward_called
                assert h.flat_param is flat_param
                with apply_lock, torch.cuda.stream(fsdp_stream):
                    # We invoke `prepare_gradient_for_optim` earlier than usual.
                    # We also need to prevent the later "normal" invocation,
                    # otherwise the double call will trigger FSDP asserts.
                    prepare_gradient = h.prepare_gradient_for_optim
                    assert hasattr(prepare_gradient, "__func__"), prepare_gradient
                    assert prepare_gradient.__func__ is FlatParamHandle.prepare_gradient_for_optim
                    prepare_gradient()
                    h.prepare_gradient_for_optim = _no_op

                    maybe_step(flat_param._params or (), h._clear_grads_if_needed)

            hook = functools.partial(_opt_hook, h, flat_param)
            hook_handles.append(fsdp_acc_grad.register_hook(hook))

        yield

    finally:
        # Non-FSDP parameters won't have a grad hook, so handle them here.
        with apply_lock:
            maybe_step(remaining)

        # Unregister the grad hooks.
        for hook_handle in hook_handles:
            hook_handle.remove()

        # And lastly back out the handle monkey patches.
        for h in param_handles:
            if h.prepare_gradient_for_optim is _no_op:
                del h.prepare_gradient_for_optim
