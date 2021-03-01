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
from typing import Any, Callable, TYPE_CHECKING, Union

import torch

from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import WarningCache

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from pytorch_lightning.core.lightning import LightningModule

warning_cache = WarningCache()


class DeepSpeedPrecisionPlugin(PrecisionPlugin):

    def __init__(self, precision: int) -> None:
        super().__init__()
        self.precision = precision

    def pre_optimizer_step(
        self,
        pl_module: 'LightningModule',
        optimizer: 'Optimizer',
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        deepspeed_engine = pl_module.trainer.model
        # DeepSpeed not support closures.
        lambda_closure()

        if not pl_module.automatic_optimization:
            pl_module.trainer.call_hook("on_after_backward")

        deepspeed_engine.step()

        return False

    def backward(
        self,
        model: 'LightningModule',
        closure_loss: torch.Tensor,
        optimizer: 'Optimizer',
        opt_idx: int,
        should_accumulate: bool,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if is_overridden('backward', model):
            warning_cache.warn(
                "Overridden backward hook in the LightningModule will be ignored since DeepSpeed handles"
                "backward logic outside of the LightningModule"
            )
        # todo: hack around for deepspeed engine to call backward
        deepspeed_engine = model.trainer.model
        deepspeed_engine.backward(closure_loss, *args, **kwargs)
        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss

    def clip_gradients(self, optimizer: 'Optimizer', clip_val: Union[int, float], norm_type: float = 2.0) -> None:
        """
        DeepSpeed handles clipping gradients via the training type plugin.
        """
        pass
