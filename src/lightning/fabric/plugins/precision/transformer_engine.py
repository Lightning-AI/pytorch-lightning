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
import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, ContextManager, Literal, Mapping, Optional, Union

import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import (
    _ClassReplacementContextManager,
    _convert_fp_tensor,
    _DtypeContextManager,
)
from lightning.fabric.utilities.rank_zero import rank_zero_warn

if TYPE_CHECKING:
    from transformer_engine.common.recipe import DelayedScaling

_TRANSFORMER_ENGINE_AVAILABLE = RequirementCache("transformer_engine>=0.11.0")
log = logging.getLogger(__name__)


class TransformerEnginePrecision(Precision):
    """Plugin for training with fp8 precision via nvidia's
    `Transformer Engine <https://docs.nvidia.com/deeplearning/transformer-engine>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        dtype: The weights dtype to use.
        recipe: Recipe for the DelayedScaling
            `configuration <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`__.
            In dict format or the dataclass format.
        replace_layers: Whether to replace ``Linear`` and ``LayerNorm`` layers automatically with their Transformer
            Engine alternatives. Note that they don't subclass the torch equivalents so checks like
            ``isinstance(l, torch.nn.Linear)`` will not pass.

    .. note::

        Support for FP8 in the linear layers with this plugin is currently limited to tensors
        with shapes where the dimensions are divisible by 8 and 16 respectively. You might want to add padding to your
        inputs to conform to this restriction.

    """

    precision: Literal["transformer-engine", "transformer-engine-float16"] = "transformer-engine"

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        replace_layers: Optional[bool] = None,
    ) -> None:
        if not _TRANSFORMER_ENGINE_AVAILABLE:
            raise ModuleNotFoundError(str(_TRANSFORMER_ENGINE_AVAILABLE))
        from transformer_engine.common.recipe import DelayedScaling

        if recipe is None:
            recipe = DelayedScaling()
        elif isinstance(recipe, Mapping):
            recipe = dict(recipe)  # copy
            if "fp8_format" in recipe:
                from transformer_engine.common.recipe import Format

                recipe["fp8_format"] = getattr(Format, recipe["fp8_format"])
            recipe = DelayedScaling(**recipe)

        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype
        self.recipe = recipe
        self.replace_layers = replace_layers

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # avoid converting if any is found. assume the user took care of it
        if self.replace_layers and not any("transformer_engine.pytorch" in m.__module__ for m in module.modules()):
            _convert_layers(module)
        module = module.to(dtype=self.dtype)
        return module

    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(self.dtype)

    def module_init_context(self) -> ContextManager:
        dtype_ctx = self.tensor_init_context()
        stack = ExitStack()
        if self.replace_layers:
            import transformer_engine.pytorch as te

            context_manager = _ClassReplacementContextManager(
                {
                    "torch.nn.Linear": te.Linear,
                    "torch.nn.LayerNorm": te.LayerNorm,
                }
            )
            stack.enter_context(context_manager)
        stack.enter_context(dtype_ctx)
        return stack

    def forward_context(self) -> ContextManager:
        dtype_ctx = _DtypeContextManager(self.dtype)
        import transformer_engine.pytorch as te

        autocast_ctx = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        stack = ExitStack()
        stack.enter_context(dtype_ctx)
        stack.enter_context(autocast_ctx)
        return stack

    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self.dtype)

    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())


def _convert_layers(module: torch.nn.Module) -> None:
    import transformer_engine.pytorch as te

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            if child.in_features % 8 != 0 or child.out_features % 16 != 0:
                # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#FP8-autocasting
                rank_zero_warn(
                    "Support for FP8 in the linear layers with this plugin is currently limited to"
                    " tensors with shapes where the dimensions are divisible by 8 and 16 respectively."
                    f" The layer {name!r} does not fit this criteria. You might want to add padding to your inputs."
                )
                continue
            has_bias = child.bias is not None
            replacement = te.Linear(child.in_features, child.out_features, bias=has_bias)
            replacement.weight.data = child.weight.data.clone()
            if has_bias:
                replacement.bias.data = child.bias.data.clone()
            log.debug(f"Replacing layer {name!r} with Transformer Engine equivalent")
            module.__setattr__(name, replacement)
        elif isinstance(child, torch.nn.LayerNorm):
            replacement = te.LayerNorm(child.normalized_shape[0], eps=child.eps)
            replacement.weight.data = child.weight.data.clone()
            replacement.bias.data = child.bias.data.clone()
            log.debug(f"Replacing layer {name!r} with Transformer Engine equivalent")
            module.__setattr__(name, replacement)
        else:
            # there are other transformer engine layers that we could convert but require fusion. full list at:
            # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
            _convert_layers(child)
