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
from typing import Any, Callable

import torch
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins.precision.xla import _PRECISION_INPUT, _PRECISION_INPUT_STR
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.plugins.precision.precision import Precision


class XLAPrecision(Precision):
    """Plugin for training with XLA.

    Args:
        precision: Full precision (32-true) or half precision (16-true, bf16-true).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT = "32-true") -> None:
        super().__init__()

        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.precision.xla import XLAPrecision as EnterpriseXLAPrecision

        self.xla_impl = EnterpriseXLAPrecision(precision)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        return self.xla_impl.optimizer_step(optimizer, model, closure, **kwargs)

    @property
    def precision(self) -> _PRECISION_INPUT_STR:
        return self.xla_impl.precision

    @precision.setter
    def precision(self, precision: _PRECISION_INPUT_STR) -> None:
        self.xla_impl.precision = precision

    @property
    def _desired_dtype(self) -> torch.dtype:
        return self.xla_impl._desired_dtype

    @_desired_dtype.setter
    def _desired_dtype(self, dtype: torch.dtype) -> None:
        self.xla_impl._desired_dtype = dtype

    @override
    def teardown(self) -> None:
        return self.xla_impl.teardown()
