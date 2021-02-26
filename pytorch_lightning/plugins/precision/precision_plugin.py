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
from typing import Any, Callable, Generator, Sequence, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.base_plugin import Plugin


class PrecisionPlugin(Plugin):
    """ Plugin handling the precision-specific parts of the training.
    The static classattributes EPSILON and precision must be overwritten in child-classes and their
    default values reflect fp32 training.
    """
    EPSILON = 1e-6
    precision = 32

    def master_params(self, optimizer: Optimizer) -> Generator[torch.Tensor, None, None]:
        """The master params of the model. Returns the plain model params here.
        Maybe different in other precision plugins.

        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    def connect(self, model: Module, optimizers: Sequence,
                lr_schedulers: Sequence) -> Tuple[Module, Sequence, Sequence]:
        """Connects this plugin to the accelerator and the training process"""
        return model, optimizers, lr_schedulers

    def backward(
        self,
        model: LightningModule,
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
        self, pl_module: LightningModule, optimizer: Optimizer, optimizer_idx: int, closure: Callable, **kwargs
    ) -> bool:
        """Hook to do something before each optimizer step."""
        return True

    def post_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        """Hook to do something after each optimizer step."""

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = float(2.0)) -> None:
        """Clips the gradients to a specific value"""
        # TODO: separate TPU case from here
        if clip_val is None:
            return

        grad_clip_val = float(clip_val)

        if grad_clip_val <= 0:
            return

        parameters = list(self.master_params(optimizer))

        max_norm = grad_clip_val

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        device = parameters[0].device

        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        eps = self.EPSILON

        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))
