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
from functools import partial
from typing import Any, Callable

import torch
from typing_extensions import get_args, override

import lightning.pytorch as pl
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins.precision.xla import _PRECISION_INPUT
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class XLAPrecision(Precision):
    """Plugin for training with XLA.

    Args:
        precision: Full precision (32-true) or half precision (16-true, bf16-true).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT = "32-true") -> None:
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
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        import torch_xla.core.xla_model as xm

        closure = partial(self._xla_wrap_closure, optimizer, closure)
        closure = partial(self._wrap_closure, model, optimizer, closure)
        closure_result = optimizer.step(closure=closure, **kwargs)
        xm.mark_step()
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            # we lack coverage here so disable this - something to explore if there's demand
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not implemented with XLA."
                " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                " requesting this feature."
            )
        return closure_result

    @override
    def teardown(self) -> None:
        os.environ.pop("XLA_USE_BF16", None)
        os.environ.pop("XLA_USE_F16", None)

    def _xla_wrap_closure(self, optimizer: Optimizable, closure: Callable[[], Any]) -> Any:
        import torch_xla.core.xla_model as xm

        closure_result = closure()
        xm.reduce_gradients(optimizer)
        return closure_result
