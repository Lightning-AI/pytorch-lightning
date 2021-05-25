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
        logging_interval: set to ``'epoch'`` or ``'step'`` to log ``lr`` of all optimizers
            at the same interval, set to ``None`` to log at individual interval
            according to the ``interval`` key of each scheduler. Defaults to ``None``.
        log_momentum: option to also log the momentum values of the optimizer, if the optimizer
            has the ``momentum`` or ``betas`` attribute. Defaults to ``False``.

    Raises:
        MisconfigurationException:
            If ``logging_interval`` is none of ``"step"``, ``"epoch"``, or ``None``.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LearningRateMonitor
        >>> lr_monitor = LearningRateMonitor(logging_interval='step')
        >>> trainer = Trainer(callbacks=[lr_monitor])

    Logging names are automatically determined based on optimizer class name.
    In case of multiple optimizers of same type, they will be named ``Adam``,
    ``Adam-1`` etc. If a optimizer has multiple parameter groups they will
    be named ``Adam/pg1``, ``Adam/pg2`` etc. To control naming, pass in a
    ``name`` keyword in the construction of the learning rate schdulers

    Example::

        def configure_optimizer(self):
            optimizer = torch.optim.Adam(...)
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
                'name': 'my_logging_name'
            }
            return [optimizer], [lr_scheduler]

    """

    def __init__(self, logging_interval: Optional[str] = None, log_momentum: bool = False):
        if logging_interval not in (None, 'step', 'epoch'):
            raise MisconfigurationException('logging_interval should be `step` or `epoch` or `None`.')

        self.logging_interval = logging_interval
        self.log_momentum = log_momentum
        self.lrs = None
        self.lr_sch_names = []

    def on_train_start(self, trainer, *args, **kwargs):
        """
        Called before training, determines unique names for all lr
        schedulers in the case of multiple of the same type or in
        the case of multiple parameter groups

        Raises:
            MisconfigurationException:
                If ``Trainer`` has no ``logger``.
        """
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use `LearningRateMonitor` callback with `Trainer` that has no logger.'
            )

        if not trainer.lr_schedulers:
            rank_zero_warn(
                'You are using `LearningRateMonitor` callback with models that'
                ' have no learning rate schedulers. Please see documentation'
                ' for `configure_optimizers` method.', RuntimeWarning
            )

        if self.log_momentum:

            def _check_no_key(key):
                return any(key not in sch['scheduler'].optimizer.defaults for sch in trainer.lr_schedulers)

            if _check_no_key('momentum') and _check_no_key('betas'):
                rank_zero_warn(
                    "You have set log_momentum=True, but some optimizers do not"
                    " have momentum. This will log a value 0 for the momentum.", RuntimeWarning
                )

        # Find names for schedulers
        names = self._find_names(trainer.lr_schedulers)

        # Initialize for storing values
        self.lrs = {name: [] for name in names}
        self.last_momentum_values = {name + "-momentum": None for name in names}

    def on_train_batch_start(self, trainer, *args, **kwargs):
        if not self._should_log(trainer):
            return

        if self.logging_interval != 'epoch':
            interval = 'step' if self.logging_interval is None else 'any'
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.global_step)

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        if self.logging_interval != 'step':
            interval = 'epoch' if self.logging_interval is None else 'any'
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.global_step)

    def _extract_stats(self, trainer, interval: str) -> Dict[str, float]:
        latest_stat = {}

        for name, scheduler in zip(self.lr_sch_names, trainer.lr_schedulers):
            if scheduler['interval'] == interval or interval == 'any':
                opt = scheduler['scheduler'].optimizer
                param_groups = opt.param_groups
                use_betas = 'betas' in opt.defaults

                for i, pg in enumerate(param_groups):
                    suffix = f'/pg{i + 1}' if len(param_groups) > 1 else ''
                    lr = self._extract_lr(param_group=pg, name=f'{name}{suffix}')
                    latest_stat.update(lr)
                    momentum = self._extract_momentum(
                        param_group=pg, name=f'{name}-momentum{suffix}', use_betas=use_betas
                    )
                    latest_stat.update(momentum)

        return latest_stat

    def _extract_lr(self, param_group, name: str) -> Dict[str, float]:
        lr = param_group.get('lr')
        self.lrs[name].append(lr)
        return {name: lr}

    def _extract_momentum(self, param_group, name: str, use_betas: bool) -> Dict[str, float]:
        if not self.log_momentum:
            return {}

        momentum = param_group.get('betas')[0] if use_betas else param_group.get('momentum', 0)
        self.last_momentum_values[name] = momentum
        return {name: momentum}

    def _find_names(self, lr_schedulers) -> List[str]:
        # Create uniqe names in the case we have multiple of the same learning
        # rate schduler + multiple parameter groups
        names = []
        for scheduler in lr_schedulers:
            sch = scheduler['scheduler']
            if scheduler['name'] is not None:
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
        return (trainer.global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop
