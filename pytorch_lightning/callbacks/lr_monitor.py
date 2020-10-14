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

r"""

Learning Rate Monitor
=====================

Monitor and logs learning rate for lr schedulers during training.

"""

from typing import Dict, List, Optional

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LearningRateMonitor(Callback):
    r"""
    Automatically monitor and logs learning rate for learning rate schedulers during training.

    Args:
        logging_interval: set to `epoch` or `step` to log `lr` of all optimizers
            at the same interval, set to `None` to log at individual interval
            according to the `interval` key of each scheduler. Defaults to ``None``.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LearningRateMonitor
        >>> lr_monitor = LearningRateMonitor(logging_interval='step')
        >>> trainer = Trainer(callbacks=[lr_monitor])

    Logging names are automatically determined based on optimizer class name.
    In case of multiple optimizers of same type, they will be named `Adam`,
    `Adam-1` etc. If a optimizer has multiple parameter groups they will
    be named `Adam/pg1`, `Adam/pg2` etc. To control naming, pass in a
    `name` keyword in the construction of the learning rate schdulers

    Example::

        def configure_optimizer(self):
            optimizer = torch.optim.Adam(...)
            lr_scheduler = {'scheduler': torch.optim.lr_schedulers.LambdaLR(optimizer, ...)
                            'name': 'my_logging_name'}
            return [optimizer], [lr_scheduler]

    """
    def __init__(self, logging_interval: Optional[str] = None):
        if logging_interval not in (None, 'step', 'epoch'):
            raise MisconfigurationException(
                'logging_interval should be `step` or `epoch` or `None`.'
            )

        self.logging_interval = logging_interval
        self.lrs = None
        self.lr_sch_names = []

    def on_train_start(self, trainer, *args, **kwargs):
        """
        Called before training, determines unique names for all lr
        schedulers in the case of multiple of the same type or in
        the case of multiple parameter groups
        """
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use LearningRateMonitor callback with Trainer that has no logger.'
            )

        if not trainer.lr_schedulers:
            rank_zero_warn(
                'You are using LearningRateMonitor callback with models that'
                ' have no learning rate schedulers. Please see documentation'
                ' for `configure_optimizers` method.', RuntimeWarning
            )

        # Find names for schedulers
        names = self._find_names(trainer.lr_schedulers)

        # Initialize for storing values
        self.lrs = {name: [] for name in names}

    def on_train_batch_start(self, trainer, *args, **kwargs):
        if not self._should_log(trainer):
            return

        if self.logging_interval != 'epoch':
            interval = 'step' if self.logging_interval is None else 'any'
            latest_stat = self._extract_lr(trainer, interval)

            if trainer.logger is not None and latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.global_step)

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        if self.logging_interval != 'step':
            interval = 'epoch' if self.logging_interval is None else 'any'
            latest_stat = self._extract_lr(trainer, interval)

            if trainer.logger is not None and latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.current_epoch)

    def _extract_lr(self, trainer, interval: str) -> Dict[str, float]:
        latest_stat = {}

        for name, scheduler in zip(self.lr_sch_names, trainer.lr_schedulers):
            if scheduler['interval'] == interval or interval == 'any':
                param_groups = scheduler['scheduler'].optimizer.param_groups
                if len(param_groups) != 1:
                    for i, pg in enumerate(param_groups):
                        lr, key = pg['lr'], f'{name}/pg{i + 1}'
                        self.lrs[key].append(lr)
                        latest_stat[key] = lr
                else:
                    self.lrs[name].append(param_groups[0]['lr'])
                    latest_stat[name] = param_groups[0]['lr']

        return latest_stat

    def _find_names(self, lr_schedulers) -> List[str]:
        # Create uniqe names in the case we have multiple of the same learning
        # rate schduler + multiple parameter groups
        names = []
        for scheduler in lr_schedulers:
            sch = scheduler['scheduler']
            if 'name' in scheduler:
                name = scheduler['name']
            else:
                opt_name = 'lr-' + sch.optimizer.__class__.__name__
                i, name = 1, opt_name

                # Multiple schduler of the same type
                while True:
                    if name not in names:
                        break
                    i, name = i + 1, f'{opt_name}-{i}'

            # Multiple param groups for the same schduler
            param_groups = sch.optimizer.param_groups

            if len(param_groups) != 1:
                for i, pg in enumerate(param_groups):
                    temp = f'{name}/pg{i + 1}'
                    names.append(temp)
            else:
                names.append(name)

            self.lr_sch_names.append(name)

        return names

    @staticmethod
    def _should_log(trainer) -> bool:
        should_log = (
            (trainer.global_step + 1) % trainer.log_every_n_steps == 0
            or trainer.should_stop
        )

        should_log = should_log and not trainer.fast_dev_run
        return should_log
