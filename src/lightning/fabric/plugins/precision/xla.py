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
import os
from typing import Any, Literal

import torch
from typing_extensions import get_args, override

from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins.precision.precision import Precision
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
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in XLA."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = precision

        if precision == "16-true":
            os.environ["XLA_USE_F16"] = "1"
            self._desired_dtype = torch.float16
        elif precision == "bf16-true":
            os.environ["XLA_USE_BF16"] = "1"
            self._desired_dtype = torch.bfloat16
        else:
            self._desired_dtype = torch.float32

    @override
    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        import torch_xla.core.xla_model as xm

        # you always want to `xm.mark_step()` after `optimizer.step` for better performance, so we set `barrier=True`
        return xm.optimizer_step(optimizer, optimizer_args=kwargs, barrier=True)

    @override
    def teardown(self) -> None:
        os.environ.pop("XLA_USE_BF16", None)
        os.environ.pop("XLA_USE_F16", None)
