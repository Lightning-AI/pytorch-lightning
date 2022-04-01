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
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import WarningCache

if _DEEPSPEED_AVAILABLE:
    from deepspeed import DeepSpeedEngine

warning_cache = WarningCache()


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for DeepSpeed integration."""

    def __init__(self, precision: Union[str, int], amp_type: str, amp_level: Optional[str] = None) -> None:
        super().__init__()
        self.precision = precision
        self.amp_type = amp_type
        self.amp_level = amp_level

    def backward(self, model: "pl.LightningModule", closure_loss: Tensor, *args: Any, **kwargs: Any) -> None:
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        assert model.trainer is not None
        deepspeed_engine: DeepSpeedEngine = model.trainer.model
        deepspeed_engine.backward(closure_loss, *args, **kwargs)

    def _run_backward(self, tensor: Tensor, model: Optional["DeepSpeedEngine"], *args: Any, **kwargs: Any) -> None:
        if model is None:
            raise ValueError("Please provide the model as input to `backward`.")
        model.backward(tensor, *args, **kwargs)

    def optimizer_step(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"DeepSpeed and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if isinstance(model, pl.LightningModule) and model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `DeepSpeed`"
            )
        # DeepSpeed handles the optimizer step internally
        deepspeed_engine: DeepSpeedEngine
        if isinstance(model, pl.LightningModule):
            assert model.trainer is not None
            deepspeed_engine = model.trainer.model
        else:
            deepspeed_engine = model
        return deepspeed_engine.step(**kwargs)

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """DeepSpeed handles gradient clipping internally."""

    def _track_grad_norm(self, trainer: "pl.Trainer") -> None:
        if trainer.track_grad_norm == -1:
            return
        # the gradients are not available in the model due to gradient partitioning in zero stage >= 2
        warning_cache.warn(
            f"You set `Trainer(track_grad_norm={trainer.track_grad_norm!r})' but this is not supported for DeepSpeed."
            " The setting will be ignored."
        )
