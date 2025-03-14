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
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal

import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities.rank_zero import rank_zero_deprecation


class DoublePrecision(Precision):
    """Plugin for training with double (``torch.float64``) precision."""

    precision: Literal["64-true"] = "64-true"

    @override
    def convert_module(self, module: nn.Module) -> nn.Module:
        return module.double()

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return _DtypeContextManager(torch.float64)

    @override
    def module_init_context(self) -> AbstractContextManager:
        return self.tensor_init_context()

    @override
    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type.

        See: :func:`torch.set_default_dtype`

        """
        with self.tensor_init_context():
            yield

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.double)


class LightningDoublePrecisionModule(_DeviceDtypeModuleMixin, nn.Module):
    """LightningModule wrapper which converts incoming floating point data in ``*_step`` and ``forward`` to double
    (``torch.float64``) precision.

    .. deprecated:: Use :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` instead.

    Args:
        pl_module: the model to wrap

    """

    def __init__(self, pl_module: "pl.LightningModule") -> None:
        super().__init__()
        rank_zero_deprecation(
            f"The `{type(self).__name__}` is deprecated and no longer needed. Convert the inputs to the `*_step`"
            f" methods directly using `trainer.precision_plugin.convert_input(...)`."
        )
        self.module = pl_module

        # set the parameters_to_ignore from LightningModule.
        _ddp_params_and_buffers_to_ignore = getattr(pl_module, "_ddp_params_and_buffers_to_ignore", [])
        self._ddp_params_and_buffers_to_ignore = [f"module.{p}" for p in _ddp_params_and_buffers_to_ignore]

    @staticmethod
    def _move_float_tensors_to_double(collection: Any) -> Any:
        return apply_to_collection(collection, Tensor, function=_convert_fp_tensor, dst_type=torch.double)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.training_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.validation_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.test_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.predict_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )
