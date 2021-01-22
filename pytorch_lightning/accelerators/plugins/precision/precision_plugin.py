import math
from typing import Union

import torch
from torch.optim import Optimizer

from pytorch_lightning.accelerators.plugins.base_plugin import Plugin
from pytorch_lightning.core import LightningModule


class PrecisionPlugin(Plugin):
    EPSILON = 1e-6
    precision = 32

    def pre_optimizer_step(self, optimizer, optimizer_idx):
        pass

    def post_optimizer_step(self, optimizer, optimizer_idx):
        pass

    def master_params(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    def connect(self, model: torch.nn.Module, optimizers, lr_schedulers):
        return model, optimizers, lr_schedulers

    def backward(
        self,
        model: LightningModule,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ):
        automatic_optimization = model.automatic_optimization

        # do backward pass
        if automatic_optimization:
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = float(2.0)):
        # TODO: separate TPU case from here
        if clip_val is None:
            return

        grad_clip_val = float(clip_val)

        if grad_clip_val <= 0:
            return

        parameters = self.master_params(optimizer)

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
