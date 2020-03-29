import warnings
from abc import ABC
from typing import List, Tuple

import torch
from torch import optim
from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.lightning import LightningModule


class TrainerOptimizersMixin(ABC):

    def init_optimizers(
            self,
            model: LightningModule
    ) -> Tuple[List, List]:
        optimizers = model.configure_optimizers()

        if optimizers is None:
            warnings.warn('`LightningModule.configure_optimizers` is not overriden or returned `None`,'
                          'this fit will run with no optimizer', UserWarning)
            optimizers = _MockOptimizer()

            # single output, single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists, optimizer + lr schedulers
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            lr_schedulers = self.configure_schedulers(lr_schedulers)
            return optimizers, lr_schedulers

        # single list or tuple, multiple optimizer
        elif isinstance(optimizers, (list, tuple)):
            return optimizers, []

        # unknown configuration
        else:
            raise ValueError('Unknown configuration for model optimizers. Output'
                             'from model.configure_optimizers() should either be:'
                             '* single output, single torch.optim.Optimizer'
                             '* single output, list of torch.optim.Optimizer'
                             '* two outputs, first being a list of torch.optim.Optimizer',
                             'second being a list of torch.optim.lr_scheduler')

    def configure_schedulers(self, schedulers: list):
        # Convert each scheduler into dict sturcture with relevant information
        lr_schedulers = []
        default_config = {'interval': 'epoch',  # default every epoch
                          'frequency': 1,  # default every epoch/batch
                          'reduce_on_plateau': False,  # most often not ReduceLROnPlateau scheduler
                          'monitor': 'val_loss'}  # default value to monitor for ReduceLROnPlateau
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                if 'scheduler' not in scheduler:
                    raise ValueError(f'Lr scheduler should have key `scheduler`',
                                     ' with item being a lr scheduler')
                scheduler['reduce_on_plateau'] = isinstance(
                    scheduler['scheduler'], optim.lr_scheduler.ReduceLROnPlateau)

                lr_schedulers.append({**default_config, **scheduler})

            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedulers.append({**default_config, 'scheduler': scheduler,
                                      'reduce_on_plateau': True})

            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                lr_schedulers.append({**default_config, 'scheduler': scheduler})
            else:
                raise ValueError(f'Input {scheduler} to lr schedulers '
                                 'is a invalid input.')
        return lr_schedulers


class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None`
    is returned from `configure_optimizers`.
    """

    def __init__(self):
        super().__init__([torch.zeros(1)], {})

    def add_param_group(self, param_group):
        pass  # Do Nothing

    def load_state_dict(self, state_dict):
        pass  # Do Nothing

    def state_dict(self):
        return {}  # Return Empty

    def step(self, closure=None):
        if closure is not None:
            closure()

    def zero_grad(self):
        pass  # Do Nothing

    def __repr__(self):
        return 'No Optimizer'
