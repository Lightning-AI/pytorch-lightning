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
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins.precision.deepspeed import _PRECISION_INPUT, _PRECISION_INPUT_STR
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import Steppable
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities import GradClipAlgorithmType


class DeepSpeedPrecision(Precision):
    """Precision plugin for DeepSpeed integration.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

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
            DeepSpeedPrecisionTrainer as EnterpriseDeepSpeedPrecision,
        )

        self.deepspeed_precision_impl = EnterpriseDeepSpeedPrecision(outer_object=self, precision=precision)

    @override
    def convert_module(self, module: Module) -> Module:
        return self.deepspeed_precision_impl.convert_module(module=module)

    @override
    def convert_input(self, data: Any) -> Any:
        return self.deepspeed_precision_impl.convert_input(data=data)

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return self.deepspeed_precision_impl.tensor_init_context()

    @override
    def module_init_context(self) -> AbstractContextManager:
        return self.deepspeed_precision_impl.module_init_context()

    @override
    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs back-propagation.

        Args:
            tensor: the loss tensor
            model: the model to be optimized
            optimizer: ignored for DeepSpeed
            \*args: additional positional arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call
            \**kwargs: additional keyword arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call

        """
        return self.deepspeed_precision_impl.backward(tensor=tensor, model=model, optimizer=optimizer, *args, **kwargs)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        return self.deepspeed_precision_impl.optimizer_step(optimizer=optimizer, model=model, closure=closure, **kwargs)

    @override
    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        return self.deepspeed_precision_impl.clip_gradients(
            optimizer=optimizer, clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm
        )

    @property
    def precision(self) -> _PRECISION_INPUT_STR:
        return self.deepspeed_precision_impl.precision

    @precision.setter
    def precision(self, value: _PRECISION_INPUT_STR) -> None:
        self.deepspeed_precision_impl.precision = value

    @property
    def _desired_dtype(self) -> torch.dtype:
        return self.deepspeed_precision_impl._desired_dtype

    @_desired_dtype.setter
    def _desired_dtype(self, value: torch.dtype) -> None:
        self.deepspeed_precision_impl._desired_dtype = value
