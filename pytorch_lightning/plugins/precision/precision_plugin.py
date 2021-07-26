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
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.plugins.base_plugin import Plugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.types import _PARAMETERS


class PrecisionPlugin(Plugin, CheckpointHooks):
    """
    Base class for all plugins handling the precision-specific parts of the training.
    The class attribute precision must be overwritten in child classes.
    The default value reflects fp32 training.
    """

    precision: Union[str, int] = 32

    def master_params(self, optimizer: Optimizer) -> _PARAMETERS:
        """
        The master params of the model. Returns the plain model params here.
        Maybe different in other precision plugins.
        """
        for group in optimizer.param_groups:
            yield from group["params"]

    def connect(
        self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[Module, List[Optimizer], List[Any]]:
        """Connects this plugin to the accelerator and the training process"""
        return model, optimizers, lr_schedulers

    def pre_backward(self, model: "pl.LightningModule", closure_loss: Tensor) -> Tensor:
        """Run before precision plugin executes backward

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
        """
        model.trainer.call_hook("on_before_backward", closure_loss)
        return closure_loss

    def backward(
        self,
        model: "pl.LightningModule",
        closure_loss: Tensor,
        optimizer: Optional[Optimizer],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Performs the actual backpropagation

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
            optimizer: current optimizer being used. ``None`` if using manual optimization
        """
        # do backward pass
        if model is not None and isinstance(model, pl.LightningModule):
            model.backward(closure_loss, optimizer, *args, **kwargs)
        else:
            closure_loss.backward(*args, **kwargs)

    def post_backward(self, model: "pl.LightningModule", closure_loss: Tensor) -> Tensor:
        """Run after precision plugin executes backward

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
        """
        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        model.trainer.call_hook("on_after_backward")
        return closure_loss

    def pre_optimizer_step(
        self,
        model: "pl.LightningModule",
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        """Hook to do something before each optimizer step."""
        model.trainer.call_hook("on_before_optimizer_step", optimizer, optimizer_idx)
        return True

    def post_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        """Hook to do something after each optimizer step."""

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
        model: Optional[Module] = None,
    ) -> None:
        """Clips the gradients"""
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            self.clip_grad_by_value(optimizer, clip_val)
        elif gradient_clip_algorithm == GradClipAlgorithmType.NORM:
            # TODO: there should be a mechanism to set `norm_type`
            self.clip_grad_by_norm(optimizer, clip_val)

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by value"""
        parameters = self.master_params(optimizer)
        torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by norm"""
        parameters = self.master_params(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, clip_val)
