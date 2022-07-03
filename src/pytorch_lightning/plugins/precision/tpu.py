# Copyright The PyTorch Lightning team.
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
from typing import Any, Callable, Optional, Union

from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class TPUPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for TPU integration."""

    def optimizer_step(
        self,
        model: Optional[Union["pl.LightningModule", Module]],
        optimizer: Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any
    ) -> Any:
        if isinstance(model, pl.LightningModule):
            closure = partial(self._wrap_closure, model, optimizer, optimizer_idx, closure)
        closure_result = xm.optimizer_step(optimizer, optimizer_args={"closure": closure, **kwargs})
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if isinstance(model, pl.LightningModule) and model.automatic_optimization and skipped_backward:
            # we lack coverage here so disable this - something to explore if there's demand
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not implemented for TPUs."
                " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                " requesting this feature."
            )
        return closure_result
