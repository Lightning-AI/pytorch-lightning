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

from abc import ABC
from typing import List, Tuple

import torch
from torch import optim
from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn


class TrainerOptimizersMixin(ABC):

    def init_optimizers(
            self,
            model: LightningModule
    ) -> Tuple[List, List, List]:
        optim_conf = model.configure_optimizers()

        if optim_conf is None:
            rank_zero_warn('`LightningModule.configure_optimizers` returned `None`, '
                           'this fit will run with no optimizer', UserWarning)
            optim_conf = _MockOptimizer()

        # single output, single optimizer
        if isinstance(optim_conf, Optimizer):
            return [optim_conf], [], []

        # two lists, optimizer + lr schedulers
        elif isinstance(optim_conf, (list, tuple)) and len(optim_conf) == 2 \
                and isinstance(optim_conf[0], list):
            optimizers, lr_schedulers = optim_conf
            lr_schedulers = self.configure_schedulers(lr_schedulers)
            return optimizers, lr_schedulers, []

        # single dictionary
        elif isinstance(optim_conf, dict):
            optimizer = optim_conf["optimizer"]
            lr_scheduler = optim_conf.get("lr_scheduler", [])
            if lr_scheduler:
                lr_schedulers = self.configure_schedulers([lr_scheduler])
            else:
                lr_schedulers = []
            return [optimizer], lr_schedulers, []

        # multiple dictionaries
        elif isinstance(optim_conf, (list, tuple)) and isinstance(optim_conf[0], dict):
            optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
            # take only lr wif exists and ot they are defined - not None
            lr_schedulers = [
                opt_dict["lr_scheduler"] for opt_dict in optim_conf if opt_dict.get("lr_scheduler")
            ]
            # take only freq wif exists and ot they are defined - not None
            optimizer_frequencies = [
                opt_dict["frequency"] for opt_dict in optim_conf if opt_dict.get("frequency") is not None
            ]

            # clean scheduler list
            if lr_schedulers:
                lr_schedulers = self.configure_schedulers(lr_schedulers)
            # assert that if frequencies are present, they are given for all optimizers
            if optimizer_frequencies and len(optimizer_frequencies) != len(optimizers):
                raise ValueError("A frequency must be given to each optimizer.")
            return optimizers, lr_schedulers, optimizer_frequencies

        # single list or tuple, multiple optimizer
        elif isinstance(optim_conf, (list, tuple)):
            return list(optim_conf), [], []

        # unknown configuration
        else:
            raise ValueError(
                'Unknown configuration for model optimizers.'
                ' Output from `model.configure_optimizers()` should either be:'
                ' * single output, single `torch.optim.Optimizer`'
                ' * single output, list of `torch.optim.Optimizer`'
                ' * single output, a dictionary with `optimizer` key (`torch.optim.Optimizer`)'
                '    and an optional `lr_scheduler` key (`torch.optim.lr_scheduler`)'
                ' * two outputs, first being a list of `torch.optim.Optimizer` second being'
                '    a list of `torch.optim.lr_scheduler`'
                ' * multiple outputs, dictionaries as described with an optional `frequency` key (int)')

    def configure_schedulers(self, schedulers: list):
        # Convert each scheduler into dict structure with relevant information
        lr_schedulers = []
        default_config = {'interval': 'epoch',  # default every epoch
                          'frequency': 1,  # default every epoch/batch
                          'reduce_on_plateau': False,  # most often not ReduceLROnPlateau scheduler
                          'monitor': 'val_loss'}  # default value to monitor for ReduceLROnPlateau
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                if 'scheduler' not in scheduler:
                    raise ValueError('Lr scheduler should have key `scheduler`',
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

    def reinit_scheduler_properties(self, optimizers: list, schedulers: list):
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']

            for optimizer in optimizers:
                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        if (
                            mro == optim.lr_scheduler._LRScheduler
                            or mro == optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            idx = i
                            state = scheduler.state_dict()
                        else:
                            state = None

                scheduler.__class__.__mro__[idx].__init__(scheduler, optimizer)
                if state is not None:
                    scheduler.load_state_dict(state)


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
