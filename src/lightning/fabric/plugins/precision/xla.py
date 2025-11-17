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
from typing import Any, Literal

import torch
from typing_extensions import override

from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR, Precision
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import Optimizable

_PRECISION_INPUT = Literal["32-true", "16-true", "bf16-true"]


class XLAPrecision(Precision):
    """Plugin for training with XLA.

    Args:
        precision: Full precision (32-true) or half precision (16-true, bf16-true).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.precision.xla import XLAPrecision as EnterpriseXLAPrecision

        self.xla_impl = EnterpriseXLAPrecision(precision=precision)

    @override
    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        return self.xla_impl.optimizer_step(optimizer, **kwargs)

    @override
    def teardown(self) -> None:
        return self.xla_impl.teardown()

    @property
    def _desired_dtype(self) -> torch.dtype:
        return self.xla_impl._desired_dtype

    @_desired_dtype.setter
    def _desired_dtype(self, dtype: torch.dtype) -> None:
        self.xla_impl._desired_dtype = dtype

    @property
    def precision(self) -> _PRECISION_INPUT_STR:
        return self.xla_impl.precision

    @precision.setter
    def precision(self, precision: _PRECISION_INPUT_STR) -> None:
        self.xla_impl.precision = precision
