import math
import sys
from abc import ABC, abstractmethod
import gc
import os
from typing import Optional

import torch
from torch import Tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import is_oom_error, garbage_collection_cuda

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
                    param_norm = p.grad.data.pow(norm_type).sum()
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

    def scale_batch_size(self,
                         model: LightningModule,
                         mode: str = 'power',
                         steps_per_iter: int = 3,
                         init_val: int = 2,
                         max_iters: int = 25):
        r"""
        Will iteratively try to find the largest batch size for a given model
        that does not give an out of memory (OOM) error.

        Args:
            model: Model to fit.

            mode: string setting the search mode. Either `power` or `binsearch`.
                If mode is `power` we keep multiplying the batch size by 2, until
                we get an OOM error. If mode is 'binsearch', we will initially
                also keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the
                batch size that failed.

            steps_per_iter: number of steps to run with a given batch size.
                Idealy 1 should be enough to test if a OOM error occurs,
                however in practise a few are needed

            init_val: initial batch size to start the search with

            max_iters: max number of increase in batch size done before
               algorithm is terminated

        """
        if mode not in ['power', 'binsearch']:
            raise ValueError('mode in method `scale_batch_size` can only be `power` or `binsearch')

        # Arguments we adjust during the batch size finder, save for restoring
        self.__scale_batch_dump_params()

        self.__scale_batch_reset_params(model, steps_per_iter)

        # Save initial model, that is loaded after batch size is found
        save_path = os.path.join(self.default_root_dir, 'temp_model.ckpt')
        self.save_checkpoint(str(save_path))

        # Initially we just double in size until an OOM is encountered
        new_size = _adjust_batch_size(self, value=init_val)  # initially set to init_val
        high = None
        count = 0
        while True:
            garbage_collection_cuda()
            self.global_step = 0  # reset after each try
            try:
                # Try fit
                self.fit(model)
                count += 1
                if count > max_iters:
                    break

                # Double in size
                low = new_size
                new_size = _adjust_batch_size(self, factor=2.0, string='succeeded')
            except RuntimeError as exception:
                # Only these errors should trigger an adjustment
                if is_oom_error(exception):
                    # If we fail in power mode, half the size and return
                    garbage_collection_cuda()
                    high = new_size
                    if mode != 'binsearch':
                        new_size = _adjust_batch_size(self, factor=0.5, string='failed')
                    break
                else:
                    raise  # some other error not memory related

        # If in binsearch mode, further refine the search for optimal batch size
        if mode == 'binsearch':
            while True:
                garbage_collection_cuda()
                self.global_step = 0  # reset after each try
                try:
                    # Try fit
                    self.fit(model)
                    count += 1
                    if count > max_iters:
                        break

                    # Adjust batch size
                    low = new_size
                    midval = (high + low) // 2
                    new_size = _adjust_batch_size(self, value=midval, string='succeeded')
                except RuntimeError as exception:
                    # Only these errors should trigger an adjustment
                    if is_oom_error(exception):
                        garbage_collection_cuda()
                        high = new_size
                        if high - low <= 1:
                            break
                        midval = (high + low) // 2
                        new_size = _adjust_batch_size(self, value=midval, string='failed')
                    else:
                        raise  # some other error not memory related

        garbage_collection_cuda()
        log.info(f'Finished batch size finder, will continue with full run using batch size {new_size}')

        # Restore initial state of model
        self.restore(str(save_path), on_gpu=self.on_gpu)
        os.remove(save_path)

        # Finish by resetting variables so trainer is ready to fit model
        self.__scale_batch_restore_params()

        return new_size

    def __scale_batch_dump_params(self):
        # Prevent going into infinite loop
        self.__dumped_params = {
            'max_steps': self.max_steps,
            'weights_summary': self.weights_summary,
            'logger': self.logger,
            'callbacks': self.callbacks,
            'checkpoint_callback': self.checkpoint_callback,
            'auto_scale_batch_size': self.auto_scale_batch_size,
            'optimizers': self.optimizers,
            'schedulers': self.schedulers,
            'model': self.model,
        }

    def __scale_batch_reset_params(self, model, steps_per_iter):
        self.auto_scale_batch_size = False  # prevent recursion
        self.max_steps = steps_per_iter  # take few steps
        self.weights_summary = None  # not needed before full run
        self.logger = None  # not needed before full run
        self.callbacks = []  # not needed before full run
        self.checkpoint_callback = False  # required for saving
        self.optimizers, self.schedulers = [], []  # required for saving
        self.model = model  # required for saving

    def __scale_batch_restore_params(self):
        self.max_steps = self.__dumped_params['max_steps']
        self.weights_summary = self.__dumped_params['weights_summary']
        self.logger = self.__dumped_params['logger']
        self.callbacks = self.__dumped_params['callbacks']
        self.checkpoint_callback = self.__dumped_params['checkpoint_callback']
        self.auto_scale_batch_size = self.__dumped_params['auto_scale_batch_size']
        self.optimizers = self.__dumped_params['optimizers']
        self.schedulers = self.__dumped_params['schedulers']
        self.model = self.__dumped_params['model']
        del self.__dumped_params


def _adjust_batch_size(trainer,
                       factor: float = 1.0,
                       value: Optional[int] = None,
                       string: str = None):
    """ Function for adjusting the batch size. It is expected that the user
        has provided a model that has a hparam field called `batch_size` i.e.
        `model.hparams.batch_size` should exist.

    Args:
        trainer: instance of pytorch_lightning.Trainer

        factor: value which the old batch size is multiplied by to get the
            new batch size

        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case

        string: either `succeeded` or `failed`. Used purely for logging

    """
    trainer_arg = 'batch_size'

    model = trainer.get_model()

    if hasattr(model.hparams, trainer_arg):
        batch_size = getattr(model.hparams, trainer_arg)
        if value:
            setattr(model.hparams, trainer_arg, value)
            new_size = value
            if string:
                log.info(f'Batch size {batch_size} {string}, trying batch size {new_size}')
        else:
            if batch_size > 1:
                new_size = int(batch_size * factor)
                if string:
                    log.info(f'Batch size {batch_size} {string}, trying batch size {new_size}')
                setattr(model.hparams, trainer_arg, new_size)
            else:
                raise ValueError('Could not reduce batch size any further')
    else:
        raise MisconfigurationException(
            f'Field {trainer_arg} not found in model.hparams')
    return new_size
