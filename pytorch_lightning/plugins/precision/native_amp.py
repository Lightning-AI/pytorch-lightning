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
from typing import Any, Callable, Dict, Generator

import torch
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import _NATIVE_AMP_AVAILABLE, AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class NativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    """ Plugin for native mixed precision training with :mod:`torch.cuda.amp`."""

    def __init__(self) -> None:
        super().__init__()
        if not _NATIVE_AMP_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for native AMP but your PyTorch version does not support it."
                " Consider upgrading with `pip install torch>=1.6`."
            )

        self.backend = AMPType.NATIVE
        self.scaler = torch.cuda.amp.GradScaler()

    def backward(
        self,
        model: 'pl.LightningModule',
        closure_loss: torch.Tensor,
        optimizer: Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args: Any,
        **kwargs: Any,
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
        self,
        pl_module: 'pl.LightningModule',
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
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
    def train_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context"""
        with torch.cuda.amp.autocast():
            yield

    @contextmanager
    def val_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context"""
        with torch.cuda.amp.autocast():
            yield

    @contextmanager
    def test_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context"""
        with torch.cuda.amp.autocast():
            yield

    @contextmanager
    def predict_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context"""
        with torch.cuda.amp.autocast():
            yield

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "native_amp_scaling_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["native_amp_scaling_state"])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["native_amp_scaling_state"] = self.scaler.state_dict()
