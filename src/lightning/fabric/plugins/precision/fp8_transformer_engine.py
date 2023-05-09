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
from contextlib import contextmanager
from typing import Any, Dict, Generator, Literal, Optional, TYPE_CHECKING, Union

from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.plugins.precision.precision import Precision

_TRANSFORMER_ENGINE_AVAILABLE = RequirementCache("transformer_engine")

if TYPE_CHECKING and _TRANSFORMER_ENGINE_AVAILABLE:
    from transformer_engine.common.recipe import DelayedScaling
else:
    DelayedScaling = None


class Fp8TransformerEnginePrecision(Precision):
    """Plugin for training with fp8 precision via nvidia's `Transformer Engine
    <https://docs.nvidia.com/deeplearning/transformer-engine`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        precision: The precision
        recipe: Recipe for the DelayedScaling
            `configuration <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transform
            er_engine.common.recipe.DelayedScaling`__. In dict format or the dataclass format.
    """

    precision: Literal["8-mixed-transformer-engine"] = "8-mixed-transformer-engine"

    def __init__(
        self,
        recipe: Optional[Union[Dict[str, Any], "DelayedScaling"]] = None,
    ) -> None:
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

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        import transformer_engine.pytorch as te

        with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            yield
