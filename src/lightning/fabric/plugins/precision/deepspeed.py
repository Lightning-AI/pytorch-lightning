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
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR, Precision
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import Steppable

if TYPE_CHECKING:
    from deepspeed import DeepSpeedEngine

_PRECISION_INPUT = Literal["32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"]


class DeepSpeedPrecision(Precision):
    """Precision plugin for DeepSpeed integration.

    Args:
        precision: Full precision (32-true), half precision (16-true, bf16-true) or
            mixed precision (16-mixed, bf16-mixed).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.precision.deepspeed import (
            DeepSpeedPrecisionFabric as EnterpriseDeepSpeedPrecision,
        )

        self.deepspeed_impl = EnterpriseDeepSpeedPrecision(precision=precision)

    @override
    def convert_module(self, module: Module) -> Module:
        return self.deepspeed_impl.convert_module(module)

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return self.deepspeed_impl.tensor_init_context()

    @override
    def module_init_context(self) -> AbstractContextManager:
        return self.deepspeed_impl.module_init_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return self.deepspeed_impl.convert_input(data)

    @override
    def convert_output(self, data: Any) -> Any:
        return self.deepspeed_impl.convert_output(data)

    @override
    def backward(self, tensor: Tensor, model: "DeepSpeedEngine", *args: Any, **kwargs: Any) -> None:
        return self.deepspeed_impl.backward(tensor, model, *args, **kwargs)

    @override
    def optimizer_step(
        self,
        optimizer: Steppable,
        **kwargs: Any,
    ) -> Any:
        return self.deepspeed_impl.optimizer_step(optimizer, **kwargs)

    @property
    def precision(self) -> _PRECISION_INPUT_STR:
        return self.deepspeed_impl.precision

    @precision.setter
    def precision(self, precision: _PRECISION_INPUT_STR) -> None:
        self.deepspeed_impl.precision = precision

    @property
    def _desired_dtype(self) -> torch.dtype:
        return self.deepspeed_impl._desired_dtype

    @_desired_dtype.setter
    def _desired_dtype(self, dtype: torch.dtype) -> None:
        self.deepspeed_impl._desired_dtype = dtype
