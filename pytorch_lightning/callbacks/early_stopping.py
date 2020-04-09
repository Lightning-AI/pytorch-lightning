r"""
Early Stopping
==============

Stop training when a monitored quantity has stopped improving.

"""

import numpy as np

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn


class EarlyStopping(Callback):
    r"""

    Args:
        monitor: quantity to be monitored. Default: ``'val_loss'``.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0``.
        patience: number of epochs with no improvement
            after which training will be stopped. Default: ``0``.
        verbose: verbosity mode. Default: ``False``.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: ``'auto'``.
        strict: whether to crash the training if `monitor` is
            not found in the metrics. Default: ``True``.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import EarlyStopping
        >>> early_stopping = EarlyStopping('val_loss')
        >>> trainer = Trainer(early_stop_callback=early_stopping)
    """

    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0.0, patience: int = 0,
                 verbose: bool = False, mode: str = 'auto', strict: bool = True):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        mode_dict = {
            'min': np.less,
            'max': np.greater,
            'auto': np.greater if 'acc' in self.monitor else np.less
        }

        if mode not in mode_dict:
            if self.verbose > 0:
                log.info(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        self.monitor_op = mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == np.greater else -1

    def check_metrics(self, logs):
        monitor_val = logs.get(self.monitor)
        error_msg = (f'Early stopping conditioned on metric `{self.monitor}`'
                     f' which is not available. Available metrics are:'
                     f' `{"`, `".join(list(logs.keys()))}`')

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, RuntimeWarning)

            return False

        return True

    def on_train_start(self, trainer, pl_module):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        stop_training = False
        if not self.check_metrics(logs):
            return stop_training

        current = logs.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                stop_training = True
                self.on_train_end(trainer, pl_module)

        return stop_training

    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0 and self.verbose > 0:
            rank_zero_warn('Displayed epoch numbers by `EarlyStopping` start from "1" until v0.6.x,'
                           ' but will start from "0" in v0.8.0.', DeprecationWarning)
            log.info(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')
