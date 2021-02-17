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
from contextlib import contextmanager
from typing import Callable, Generator

import torch
from torch.optim import LBFGS, Optimizer

from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import _NATIVE_AMP_AVAILABLE, AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _NATIVE_AMP_AVAILABLE:
    from torch.cuda.amp import autocast
else:
    autocast = None


class NativeMixedPrecisionPlugin(MixedPrecisionPlugin):

    def __init__(self):
        self.backend = AMPType.NATIVE
        self.scaler = torch.cuda.amp.GradScaler()

    def backward(
        self,
        model: LightningModule,
        closure_loss: torch.Tensor,
        optimizer: Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """performs the actual backpropagation

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
            optimizer: the optimizer to perform the step lateron
            opt_idx: the optimizer's index
            should_accumulate: whether to accumulate gradients or not

        """
        closure_loss = self.scaler.scale(closure_loss)

        closure_loss = super().backward(model, closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs)

        # unscale gradient to allow analyze within `on_after_backward`
        if not should_accumulate and model.automatic_optimization:
            self.scaler.unscale_(optimizer)

        return closure_loss

    def pre_optimizer_step(
        self, pl_module: LightningModule, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs
    ) -> bool:
        """always called before the optimizer step.
        Checks that the optimizer is not LBFGS, as this one is not supported by native amp
        """
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"native PyTorch amp and lbfgs are not compatible (optimizer {optimizer_idx})."
                " To request, please file a Github issue in PyTorch and tag @mcarilli"
            )
        lambda_closure()

        if not pl_module.automatic_optimization:
            self.scaler.unscale_(optimizer)
            pl_module.trainer.call_hook("on_after_backward")

        return False

    def post_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        """Updates the GradScaler"""
        self.scaler.step(optimizer)
        self.scaler.update()

    @contextmanager
    def train_step_context(self) -> Generator[autocast, None, None]:
        """Enable autocast context"""
        yield torch.cuda.amp.autocast()
