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
from functools import partial
from typing import Any, Callable

import pytorch_lightning as pl
from lightning_fabric.accelerators.tpu import _XLA_AVAILABLE
from lightning_fabric.utilities.types import Optimizable
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TPUPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for TPU integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    def _tpu_wrap_closure(self, optimizer: Optimizable, closure: Callable[[], Any]) -> Any:
        import torch_xla.core.xla_model as xm

        closure_result = closure()
        xm.reduce_gradients(optimizer)
        return closure_result

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        import torch_xla.core.xla_model as xm

        closure = partial(self._tpu_wrap_closure, optimizer, closure)
        closure = partial(self._wrap_closure, model, optimizer, optimizer_idx, closure)
        closure_result = optimizer.step(closure=closure, **kwargs)
        xm.mark_step()
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            # we lack coverage here so disable this - something to explore if there's demand
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not implemented for TPUs."
                " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                " requesting this feature."
            )
        return closure_result
