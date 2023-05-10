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
from contextlib import contextmanager
from typing import Any, Dict, Generator, Literal, Optional, TYPE_CHECKING, Union

import torch
from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.utilities.rank_zero import rank_zero_warn

_TRANSFORMER_ENGINE_AVAILABLE = RequirementCache("transformer_engine")

if TYPE_CHECKING and _TRANSFORMER_ENGINE_AVAILABLE:
    from transformer_engine.common.recipe import DelayedScaling
else:
    DelayedScaling = None


log = logging.getLogger(__name__)


class Fp8TransformerEnginePrecision(Precision):
    """Plugin for training with fp8 precision via nvidia's `Transformer Engine
    <https://docs.nvidia.com/deeplearning/transformer-engine`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        recipe: Recipe for the DelayedScaling
            `configuration <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transform
            er_engine.common.recipe.DelayedScaling`__. In dict format or the dataclass format.
        replace_layers: Whether to replace ``Linear`` and ``LayerNorm`` layers automatically with their Transformer
            Engine alternatives.
    """

    precision: Literal["8-mixed-transformer-engine"] = "8-mixed-transformer-engine"

    def __init__(
        self, recipe: Optional[Union[Dict[str, Any], "DelayedScaling"]] = None, replace_layers: bool = True
    ) -> None:
        if not torch.cuda.get_device_capability() >= (8, 1):
            raise NotImplementedError("Your CUDA device does not support fp8 mixed precision.")
        if not _TRANSFORMER_ENGINE_AVAILABLE:
            raise ModuleNotFoundError(str(_TRANSFORMER_ENGINE_AVAILABLE))
        from transformer_engine.common.recipe import DelayedScaling

        if recipe is None:
            recipe = DelayedScaling()
        elif isinstance(recipe, dict):
            if "fp8_format" in recipe:
                from transformer_engine.common.recipe import Format

                recipe["fp8_format"] = getattr(Format, recipe["fp8_format"])
            recipe = DelayedScaling(**recipe)

        self.recipe = recipe
        self.replace_layers = replace_layers

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        import transformer_engine.pytorch as te

        with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            yield

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # avoid converting if any is found. assume the user took care of it
        if self.replace_layers and not any("transformer_engine" in m.__module__ for m in module.modules()):
            _convert_layers(module)
        return module
        # TODO: should we un-convert on teardown?


def _convert_layers(module: torch.nn.Module) -> None:
    import transformer_engine.pytorch as te

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#FP8-autocasting
                rank_zero_warn(
                    "Support for FP8 in the linear layers with `precision='8-mixed'` is currently limited to tensors"
                    f" with shapes where both dimensions are divisible by 16. The layer {name!r} does not fit this"
                    " criteria.You might want to add padding to your inputs."
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
            # there are other transformer engine layers that we could convert but are more niche. full list at:
            # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
            _convert_layers(module)
