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
from typing import Any, Dict

import numpy as np
import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EarlyStopping(Callback):
    r"""
    Monitor a metric and stop training when it stops improving.

    Args:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no improvement.
        patience: number of validation checks with no improvement
            after which training will be stopped. Under the default configuration, one validation check happens after
            every training epoch. However, the frequency of validation can be modified by setting various parameters on
            the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

            .. note::

                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.

        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantity
            monitored has stopped decreasing and in ``'max'`` mode it will stop when the quantity
            monitored has stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the validation metrics.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"`` or ``"max"``.
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
        mode: str = 'min',
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

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

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

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'best_score': self.best_score,
            'patience': self.patience
        }

    def on_load_checkpoint(self, callback_state: Dict[str, Any]):
        self.wait_count = callback_state['wait_count']
        self.stopped_epoch = callback_state['stopped_epoch']
        self.best_score = callback_state['best_score']
        self.patience = callback_state['patience']

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
        else:
            self.wait_count += 1

            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_early_stopping_decision(trainer.should_stop)
