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
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

if TYPE_CHECKING:
    from transformer_engine.common.recipe import DelayedScaling


class TransformerEnginePrecision(Precision):
    """Plugin for training with fp8 precision via nvidia's
    `Transformer Engine <https://docs.nvidia.com/deeplearning/transformer-engine>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        weights_dtype: The weights dtype to use.
        recipe: Recipe for the DelayedScaling
            `configuration <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`__.
            In dict format or the dataclass format.
        replace_layers: Whether to replace ``Linear`` and ``LayerNorm`` layers automatically with their Transformer
            Engine alternatives. Note that they don't subclass the torch equivalents so checks like
            ``isinstance(l, torch.nn.Linear)`` will not pass.
        fallback_compute_dtype: The compute dtype to use for operations that don't support fp8 autocast. Defaults to the
            same as ``weights_dtype``.

    .. note::

        Support for FP8 in the linear layers with this plugin is currently limited to tensors
        with shapes where the dimensions are divisible by 8 and 16 respectively. You might want to add padding to your
        inputs to conform to this restriction.

    """

    precision: Literal["transformer-engine", "transformer-engine-float16"] = "transformer-engine"

    def __init__(
        self,
        *,
        weights_dtype: torch.dtype,
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        replace_layers: Optional[bool] = None,
        fallback_compute_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.precision.transformer_engine import (
            TransformerEnginePrecision as EnterpriseTransformerEnginePrecision,
        )

        self.transformer_engine_impl = EnterpriseTransformerEnginePrecision(
            weights_dtype=weights_dtype,
            recipe=recipe,
            replace_layers=replace_layers,
            fallback_compute_dtype=fallback_compute_dtype,
        )

    @property
    def weights_dtype(self) -> torch.dtype:
        return self.transformer_engine_impl.weights_dtype

    @weights_dtype.setter
    def weights_dtype(self, value: torch.dtype) -> None:
        self.transformer_engine_impl.weights_dtype = value

    @property
    def recipe(self) -> Union[Mapping[str, Any], "DelayedScaling"]:
        return self.transformer_engine_impl.recipe

    @recipe.setter
    def recipe(self, value: Union[Mapping[str, Any], "DelayedScaling"]) -> None:
        self.transformer_engine_impl.recipe = value

    @property
    def replace_layers(self) -> bool:
        return self.transformer_engine_impl.replace_layers

    @replace_layers.setter
    def replace_layers(self, value: bool) -> None:
        self.transformer_engine_impl.replace_layers = value

    @property
    def fallback_compute_dtype(self) -> torch.dtype:
        return self.transformer_engine_impl.fallback_compute_dtype

    @fallback_compute_dtype.setter
    def fallback_compute_dtype(self, value: torch.dtype) -> None:
        self.transformer_engine_impl.fallback_compute_dtype = value

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return self.transformer_engine_impl.convert_module(module)

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return self.transformer_engine_impl.tensor_init_context()

    @override
    def module_init_context(self) -> AbstractContextManager:
        return self.transformer_engine_impl.module_init_context()

    @override
    def forward_context(self) -> AbstractContextManager:
        return self.transformer_engine_impl.forward_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return self.transformer_engine_impl.convert_input(data)

    @override
    def convert_output(self, data: Any) -> Any:
        return self.transformer_engine_impl.convert_output(data)

    @property
    def _desired_dtype(self) -> torch.dtype:
        return self.transformer_engine_impl._desired_dtype

    @_desired_dtype.setter
    def _desired_dtype(self, dtype: torch.dtype) -> None:
        self.transformer_engine_impl._desired_dtype = dtype
