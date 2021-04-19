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
import math
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.base_plugin import Plugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.types import _PARAMETERS


class PrecisionPlugin(Plugin):
    """
    Base class for all plugins handling the precision-specific parts of the training.
    The static classattributes EPSILON and precision must be overwritten in child-classes and their
    default values reflect fp32 training.
    """
    EPSILON: float = 1e-6
    precision: Union[str, int] = 32

    def master_params(self, optimizer: Optimizer) -> _PARAMETERS:
        """
        The master params of the model. Returns the plain model params here.
        Maybe different in other precision plugins.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    def connect(
        self,
        model: Module,
        optimizers: List[Optimizer],
        lr_schedulers: List[Any],
    ) -> Tuple[Module, List[Optimizer], List[Any]]:
        """Connects this plugin to the accelerator and the training process"""
        return model, optimizers, lr_schedulers

    def backward(
        self,
        model: 'pl.LightningModule',
        closure_loss: Tensor,
        optimizer: Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """performs the actual backpropagation

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
            optimizer: the optimizer to perform the step lateron
            opt_idx: the optimizer's index
            should_accumulate: whether to accumulate gradients or not

        """
        automatic_optimization = model.automatic_optimization

        # do backward pass
        if automatic_optimization:
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss

    def pre_optimizer_step(
        self,
        pl_module: 'pl.LightningModule',
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        """Hook to do something before each optimizer step."""
        return True

    def post_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        """Hook to do something after each optimizer step."""

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
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
            self.clip_grad_by_norm(optimizer, clip_val, eps=self.EPSILON)

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by value"""
        parameters = self.master_params(optimizer)
        torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)

    def clip_grad_by_norm(
        self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = 2.0, eps: float = 1e-6
    ) -> None:
        """Clip gradients by norm"""
        parameters = self.master_params(optimizer)

        # TODO: replace this with torch.nn.clip_grad_norm_
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        device = parameters[0].device

        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        clip_coef = torch.tensor(clip_val, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))
