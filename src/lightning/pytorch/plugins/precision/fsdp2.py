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
from typing import Any

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from typing_extensions import get_args, override

from lightning.fabric.plugins.precision.fsdp import _PRECISION_INPUT
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class FSDP2Precision(Precision):
    """Precision plugin for training with FSDP2 (Fully Sharded Data Parallel v2).

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        precision: Full precision (32-true), half precision (16-true, bf16-true) or
            mixed precision (16-mixed, bf16-mixed).
        scaler: An optional :class:`torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler` to use.

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT, scaler: Any = None) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in FSDP."
                f" `precision` must be one of: {supported_precision}."
            )

        if scaler is not None:
            raise ValueError(
                f"`scaler` is not supported in `{self.__class__.__name__}`, found {scaler}."
                "Use `mixed-precision policy` instead to configure the scaler."
            )

        if "mixed" in precision:
            raise ValueError(
                f"`precision={precision!r}` is not supported in `{self.__class__.__name__}`."
                "Only `true` precision is supported."
                "Use `mixed-precision policy (mp_policy)` instead to configure mixed precision."
            )

        self.precision = precision

        precision_to_type = {
            "bf16-true": torch.bfloat16,
            "16-true": torch.float16,
            "32-true": torch.float32,
        }
        self._desired_input_dtype = precision_to_type[self.precision]

    @override
    def convert_module(self, module: Module) -> Module:
        if "true" in self.precision:
            return module.to(dtype=self._desired_input_dtype)
        return module

    @override
    def clip_grad_by_norm(self, *_: Any, **__: Any) -> None:
        # see https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
        # section `Gradient Clipping`, using `torch.nn.utils.clip_grad_norm_` is incorrect with FSDP.
        # To overcome this we need to call root_sharded_module.clip_grad_norm(clip_val), but we don't have a reference
        # to the root module
        raise MisconfigurationException(
            f"`gradient_clip_algorithm='norm'` is currently not supported for `{self.__class__.__name__}`"
        )

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> AbstractContextManager:
        # Use float32 for module parameter initialization to ensure numerical stability
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def forward_context(self) -> AbstractContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())
