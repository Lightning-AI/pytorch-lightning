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
from typing import Any, Callable, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for DeepSpeed integration."""

    def __init__(self, precision: int) -> None:
        super().__init__()
        self.precision = precision

    def backward(self, model: "pl.LightningModule", closure_loss: Tensor, *args: Any, **kwargs: Any) -> None:
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        deepspeed_engine = model.trainer.model
        deepspeed_engine.backward(closure_loss, *args, **kwargs)

    def _run_backward(self, tensor: Tensor, model: Module, *args: Any, **kwargs: Any) -> None:
        model.backward(tensor, *args, **kwargs)

    def pre_optimizer_step(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable[[], Any],
        **kwargs: Any,
    ) -> bool:
        """Hook to do something before each optimizer step."""
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"DeepSpeed and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        result = lambda_closure()  # DeepSpeed does not support closures
        super().pre_optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        skipped_backward = result is None
        # in manual optimization, the closure does not return a value
        if isinstance(model, pl.LightningModule) and model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `DeepSpeed`"
            )
        # DeepSpeed handles the optimizer step internally
        deepspeed_engine = model.trainer.model if isinstance(model, pl.LightningModule) else model
        deepspeed_engine.step(**kwargs)
        return False

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if clip_val is None or float(clip_val) <= 0:
            return
        raise MisconfigurationException("DeepSpeed handles gradient clipping internally.")
