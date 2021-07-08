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
from typing import Any, Callable, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """ Precision plugin for DeepSpeed integration. """

    def __init__(self, precision: int) -> None:
        super().__init__()
        self.precision = precision

    def pre_optimizer_step(
        self,
        pl_module: 'pl.LightningModule',
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        # DeepSpeed not support closures.
        lambda_closure()
        deepspeed_engine = pl_module.trainer.model
        deepspeed_engine.step()
        return False

    def backward(
        self,
        model: 'pl.LightningModule',
        closure_loss: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if is_overridden('backward', model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        # todo: hack around for deepspeed engine to call backward
        deepspeed_engine = model.trainer.model

        # hook
        model.trainer.call_hook("on_before_backward", closure_loss)

        deepspeed_engine.backward(closure_loss, *args, **kwargs)

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
        model: Optional[Module] = None,
    ) -> None:
        """
        DeepSpeed handles clipping gradients internally via the training type plugin.
        """
        pass
