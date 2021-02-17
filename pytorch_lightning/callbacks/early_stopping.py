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
Early Stopping
^^^^^^^^^^^^^^

Monitor a metric and stop training when it stops improving.

"""

import numpy as np
import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EarlyStopping(Callback):
    r"""
    Monitor a metric and stop training when it stops improving.

    Args:
        monitor: quantity to be monitored. Default: ``'early_stop_on'``.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0.0``.
        patience: number of validation epochs with no improvement
            after which training will be stopped. Default: ``3``.
        verbose: verbosity mode. Default: ``False``.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.

            .. warning::
               Setting ``mode='auto'`` has been deprecated in v1.1 and will be removed in v1.3.

        strict: whether to crash the training if `monitor` is
            not found in the validation metrics. Default: ``True``.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"``, ``"max"``, and ``"auto"``.
        RuntimeError:
            If the metric ``monitor`` is not available.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import EarlyStopping
        >>> early_stopping = EarlyStopping('val_loss')
        >>> trainer = Trainer(callbacks=[early_stopping])
    """
    mode_dict = {
        'min': torch.lt,
        'max': torch.gt,
    }

    def __init__(
        self,
        monitor: str = 'early_stop_on',
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = 'auto',
        strict: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta
        self.wait_count = 0
        self.stopped_epoch = 0
        self.mode = mode
        self.warned_result_obj = False

        self.__init_monitor_mode()

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def __init_monitor_mode(self):
        if self.mode not in self.mode_dict and self.mode != 'auto':
            raise MisconfigurationException(f"`mode` can be auto, {', '.join(self.mode_dict.keys())}, got {self.mode}")

        # TODO: Update with MisconfigurationException when auto mode is removed in v1.3
        if self.mode == 'auto':
            rank_zero_warn(
                "mode='auto' is deprecated in v1.1 and will be removed in v1.3."
                " Default value for mode with be 'min' in v1.3.", DeprecationWarning
            )

            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.mode = 'max'
            else:
                self.mode = 'min'

            if self.verbose > 0:
                rank_zero_info(f'EarlyStopping mode set to {self.mode} for monitoring {self.monitor}.')

    def _validate_condition_metric(self, logs):
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f'Early stopping conditioned on metric `{self.monitor}` which is not available.'
            ' Pass in or modify your `EarlyStopping` callback to use any of the following:'
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]

    def on_save_checkpoint(self, trainer, pl_module):
        return {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'best_score': self.best_score,
            'patience': self.patience
        }

    def on_load_checkpoint(self, checkpointed_state):
        self.wait_count = checkpointed_state['wait_count']
        self.stopped_epoch = checkpointed_state['stopped_epoch']
        self.best_score = checkpointed_state['best_score']
        self.patience = checkpointed_state['patience']

    def on_validation_end(self, trainer, pl_module):
        if trainer.running_sanity_check:
            return

        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run  # disable early_stopping with fast_dev_run
            or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return  # short circuit if metric not present

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
            should_stop = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience

            if bool(should_stop):
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_early_stopping_decision(should_stop)
        trainer.should_stop = should_stop
