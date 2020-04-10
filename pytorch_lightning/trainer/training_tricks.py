import math
import sys
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import GradientAccumulationScheduler

EPSILON = 1e-6
EPSILON_FP16 = 1e-5


class TrainerTrainingTricksMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    gradient_clip_val: ...
    precision: ...

    @abstractmethod
    def get_model(self):
        """Warning: this is just empty shell for code implemented in other class."""

    def clip_gradients(self):
        # this code is a modification of torch.nn.utils.clip_grad_norm_
        # with TPU support based on https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
        if self.gradient_clip_val > 0:
            model = self.get_model()
            parameters = model.parameters()
            max_norm = float(self.gradient_clip_val)
            norm_type = float(2.0)
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = list(filter(lambda p: p.grad is not None, parameters))
            if norm_type == math.inf:
                total_norm = max(p.grad.data.abs().max() for p in parameters)
            else:
                device = parameters[0].device
                total_norm = torch.zeros([], device=device if parameters else None)
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type) ** norm_type
                    total_norm.add_(param_norm)
                total_norm = (total_norm ** (1. / norm_type))
            eps = EPSILON_FP16 if self.precision == 16 else EPSILON
            clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
            for p in parameters:
                p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))

    def print_nan_gradients(self) -> None:
        model = self.get_model()
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                log.info(param, param.grad)

    def detect_nan_tensors(self, loss: Tensor) -> None:
        model = self.get_model()

        # check if loss is nan
        if not torch.isfinite(loss).all():
            raise ValueError(
                'The loss returned in `training_step` is nan or inf.'
            )
        # check if a network weight is nan
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                self.print_nan_gradients()
                raise ValueError(
                    f'Detected nan and/or inf values in `{name}`.'
                    ' Check your forward pass for numerically unstable operations.'
                )

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {1: accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")
