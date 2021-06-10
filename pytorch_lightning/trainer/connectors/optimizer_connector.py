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
from typing import List, Optional
from weakref import proxy

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class OptimizerConnector:

    def __init__(self, trainer: 'pl.Trainer') -> None:
        self.trainer = proxy(trainer)

    def on_trainer_init(self) -> None:
        self.trainer.lr_schedulers = []
        self.trainer.optimizers = []
        self.trainer.optimizer_frequencies = []

    def update_learning_rates(self, interval: str, opt_indices: Optional[List[int]] = None) -> None:
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            opt_indices: indices of the optimizers to update.
        """
        if not self.trainer.lr_schedulers or not self.trainer.lightning_module.automatic_optimization:
            return

        if opt_indices is None:
            opt_indices = []

        for scheduler_idx, lr_scheduler in enumerate(self.trainer.lr_schedulers):
            if isinstance(lr_scheduler['opt_idx'], int) and lr_scheduler['opt_idx'] not in opt_indices:
                continue

            current_idx = self.trainer.train_loop.batch_idx if interval == 'step' else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency'] == 0:
                # If instance of ReduceLROnPlateau, we need a monitor
                monitor_key, monitor_val = None, None
                if lr_scheduler['reduce_on_plateau']:
                    monitor_key = lr_scheduler['monitor']
                    monitor_val = self.trainer.logger_connector.callback_metrics.get(monitor_key)
                    if monitor_val is None:
                        if lr_scheduler.get('strict', True):
                            avail_metrics = list(self.trainer.logger_connector.callback_metrics.keys())
                            raise MisconfigurationException(
                                f'ReduceLROnPlateau conditioned on metric {monitor_key}'
                                f' which is not available. Available metrics are: {avail_metrics}.'
                                ' Condition can be set using `monitor` key in lr scheduler dict'
                            )
                        rank_zero_warn(
                            f'ReduceLROnPlateau conditioned on metric {monitor_key}'
                            ' which is not available but strict is set to `False`.'
                            ' Skipping learning rate update.',
                            RuntimeWarning,
                        )
                        continue
                # update LR
                old_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']

                if lr_scheduler['reduce_on_plateau']:
                    lr_scheduler['scheduler'].step(monitor_val)
                else:
                    lr_scheduler['scheduler'].step()

                new_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']

                if self.trainer.dev_debugger.enabled:
                    self.trainer.dev_debugger.track_lr_schedulers_update(
                        self.trainer.train_loop.batch_idx,
                        interval,
                        scheduler_idx,
                        old_lr,
                        new_lr,
                        monitor_key=monitor_key,
                        monitor_val=monitor_val
                    )
