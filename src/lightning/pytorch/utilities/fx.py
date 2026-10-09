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
from typing import Any

import torch
import torch.fx
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.utilities.model_helpers import _check_mixed_imports


def from_fx(module: "pl.LightningModule", graph_module: torch.fx.GraphModule) -> "pl.LightningModule":
    """Transfers the computation of a ``torch.fx.GraphModule`` onto the given ``LightningModule``.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Model conversions built on ``torch.fx`` (e.g. PyG's ``to_hetero()``) return a ``torch.fx.GraphModule`` that
    performs the converted computation but does not inherit from the original model's class. The
    :class:`~lightning.pytorch.trainer.trainer.Trainer` rejects such an object because the Lightning code
    (``training_step()``, ``configure_optimizers()``, hooks, ...) is not available on it. This method replaces the
    ``forward``, submodules, parameters and buffers of the :class:`~lightning.pytorch.core.LightningModule` with the
    ones from the graph.

    Note:
        this method will in-place modify the ``LightningModule`` that is passed in.

    Args:
        module: The :class:`~lightning.pytorch.core.LightningModule` that ``graph_module`` was traced from.
        graph_module: The ``torch.fx.GraphModule`` whose computation replaces the one of ``module``.

    """
    if not isinstance(module, pl.LightningModule):
        _check_mixed_imports(module)
        raise TypeError(f"`module` must be a `LightningModule`, got `{type(module).__name__}`")
    if not isinstance(graph_module, torch.fx.GraphModule):
        raise TypeError(f"`graph_module` must be a `torch.fx.GraphModule`, got `{type(graph_module).__name__}`")
    if module._compiler_ctx is not None:
        raise RuntimeError("`module` is compiled with `torch.compile()`. Call `to_uncompiled()` before `from_fx()`.")
    if module._fx_ctx is not None:
        raise RuntimeError("`module` is already converted. Call `to_unfx()` before calling `from_fx()` again.")
    trainer = module._trainer
    if trainer is not None and (trainer.strategy.optimizers or trainer.strategy.model is not module):
        raise RuntimeError(
            "`from_fx()` must be called before the Trainer sets up the model, for example before `Trainer.fit()` or"
            " inside `LightningModule.configure_model()`."
        )

    ctx: dict[str, Any] = {
        "original_forward": module.forward,
        "original_modules": dict(module._modules),
        "original_parameters": dict(module._parameters),
        "original_buffers": dict(module._buffers),
        "original_non_persistent_buffers": set(module._non_persistent_buffers_set),
    }
    module._fx_ctx = ctx

    for name, submodule in graph_module._modules.items():
        _clear_attribute(module, name)
        module._modules[name] = submodule
    for name, parameter in graph_module._parameters.items():
        _clear_attribute(module, name)
        module._parameters[name] = parameter
    for name, buffer in graph_module._buffers.items():
        _clear_attribute(module, name)
        module._buffers[name] = buffer
        # torch.fx copies buffers onto the graph module as persistent, so the original registration is consulted too
        if name in graph_module._non_persistent_buffers_set or name in ctx["original_non_persistent_buffers"]:
            module._non_persistent_buffers_set.add(name)
    for node in graph_module.graph.nodes:
        if node.op == "get_attr" and not hasattr(module, node.target):
            setattr(module, node.target, getattr(graph_module, node.target))

    module.training = graph_module.training
    module.forward = graph_module.forward.__get__(module, type(module))  # type: ignore[method-assign]

    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            try:
                module.get_submodule(node.target)
            except AttributeError:
                to_unfx(module)
                raise RuntimeError(
                    f"The graph references the submodule `{node.target}` which was not found on `module`."
                ) from None
    return module


def to_unfx(module: "pl.LightningModule") -> "pl.LightningModule":
    """Restores a :class:`~lightning.pytorch.core.LightningModule` that was converted with :func:`from_fx`.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Note:
        this method will in-place modify the ``LightningModule`` that is passed in.

    """
    if module._fx_ctx is None:
        raise ValueError(
            "`module` is required to be a converted LightningModule. Found a non-converted LightningModule instead."
        )
    ctx = module._fx_ctx
    module.forward = ctx["original_forward"]  # type: ignore[method-assign]
    module._modules.clear()
    module._modules.update(ctx["original_modules"])
    module._parameters.clear()
    module._parameters.update(ctx["original_parameters"])
    module._buffers.clear()
    module._buffers.update(ctx["original_buffers"])
    module._non_persistent_buffers_set.clear()
    module._non_persistent_buffers_set.update(ctx["original_non_persistent_buffers"])
    module._fx_ctx = None
    return module


def _clear_attribute(module: nn.Module, name: str) -> None:
    module.__dict__.pop(name, None)
    module._modules.pop(name, None)
    module._parameters.pop(name, None)
    module._buffers.pop(name, None)
    module._non_persistent_buffers_set.discard(name)
