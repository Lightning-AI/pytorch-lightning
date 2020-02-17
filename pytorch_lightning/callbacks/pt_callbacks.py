"""
Callbacks
=========

Callbacks supported by Lightning
"""

import os
import shutil
import logging as log
import warnings

import numpy as np


class Callback(object):
    """Abstract base class used to build new callbacks."""

    def __init__(self):
        self._trainer = None

    def set_trainer(self, trainer):
        """Make a link to the trainer, so different things like `trainer.current_epoch`,
        `trainer.batch_idx`, `trainer.global_step` can be used."""
        self._trainer = trainer

    def on_epoch_begin(self):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self):
        """Called when the epoch ends."""
        pass

    def on_batch_begin(self):
        """Called when the training batch begins."""
        pass

    def on_batch_end(self):
        """Called when the training batch ends."""
        pass

    def on_train_begin(self):
        """Called when the train begins."""
        pass

    def on_train_end(self):
        """Called when the train ends."""
        pass

    def on_validation_begin(self):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self):
        """Called when the validation loop ends."""
        pass

    def on_test_begin(self):
        """Called when the test begins."""
        pass

    def on_test_end(self):
        """Called when the test ends."""
        pass


_NO_TRAINER_ERROR_MSG = ".set_trainer() should be called after the callback initialization"


class EarlyStopping(Callback):
    r"""
    Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str): quantity to be monitored. Default: ``'val_loss'``.
        min_delta (float): minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0``.
        patience (int): number of epochs with no improvement
            after which training will be stopped. Default: ``0``.
        verbose (bool): verbosity mode. Default: ``0``.
        mode (str): one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: ``'auto'``.
        strict (bool): whether to crash the training if `monitor` is
            not found in the metrics. Default: ``True``.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping

        early_stopping = EarlyStopping('val_loss')
        Trainer(early_stop_callback=early_stopping)
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0.0, patience=0, verbose=0, mode='auto', strict=True):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            if self.verbose > 0:
                log.info(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.on_train_begin()

    def check_metrics(self, logs):
        monitor_val = logs.get(self.monitor)
        error_msg = (f'Early stopping conditioned on metric `{self.monitor}`'
                     f' which is not available. Available metrics are:'
                     f' `{"`, `".join(list(logs.keys()))}`')

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                warnings.warn(error_msg, RuntimeWarning)

            return False

        return True

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self):
        assert self._trainer is not None, _NO_TRAINER_ERROR_MSG

        logs = self._trainer.callback_metrics
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
                self.stopped_epoch = self._trainer.current_epoch
                stop_training = True
                self.on_train_end()

        return stop_training

    def on_train_end(self):
        if self.stopped_epoch > 0 and self.verbose > 0:
            warnings.warn('Displayed epoch numbers by `EarlyStopping` start from "1" until v0.6.x,'
                          ' but will start from "0" in v0.8.0.', DeprecationWarning)
            log.info(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')


class ModelCheckpoint(Callback):
    r"""

    Save the model after every epoch.

    Args:
        filepath (str): path to save the model file.
            Can contain named formatting options to be auto-filled.

            Example::

                # save epoch and val_loss in name
                ModelCheckpoint(filepath='{epoch:02d}-{val_loss:.2f}.hdf5')
                # saves file like: /path/epoch_2-val_loss_0.2.hdf5
        monitor (str): quantity to monitor.
        verbose (bool): verbosity mode, 0 or 1.
        save_top_k (int): if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if `save_top_k == 0`, no models are saved.
            if `save_top_k == -1`, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if `save_top_k >= 2` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode (str): one of {auto, min, max}.
            If `save_top_k != 0`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only (bool): if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period (int): Interval (number of epochs) between checkpoints.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(filepath='my_path')
        Trainer(checkpoint_callback=checkpoint_callback)

        # saves checkpoints to my_path whenever 'val_loss' has a new min
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_top_k=1, save_weights_only=False,
                 mode='auto', period=1, prefix=''):
        super(ModelCheckpoint, self).__init__()
        if (
            save_top_k and
            os.path.isdir(filepath) and
            len(os.listdir(filepath)) > 0
        ):
            warnings.warn(
                f"Checkpoint directory {filepath} exists and is not empty with save_top_k != 0."
                "All files in this directory will be deleted when a checkpoint is saved!"
            )

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        os.makedirs(filepath, exist_ok=True)
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_check = 0
        self.prefix = prefix
        self.best_k_models = {}
        # {filename: monitor}
        self.kth_best_model = ''
        self.best = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                f'ModelCheckpoint mode {mode} is unknown, '
                'fallback to auto mode.', RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.kth_value = np.Inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = np.greater
            self.kth_value = -np.Inf
            self.mode = 'max'
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.kth_value = -np.Inf
                self.mode = 'max'
            else:
                self.monitor_op = np.less
                self.kth_value = np.Inf
                self.mode = 'min'

    def _del_model(self, filepath):
        dirpath = os.path.dirname(filepath)

        # make paths
        os.makedirs(dirpath, exist_ok=True)

        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    def _save_model(self, filepath):
        dirpath = os.path.dirname(filepath)

        # make paths
        os.makedirs(dirpath, exist_ok=True)

        # delegate the saving to the model
        self.save_function(filepath)

    def check_monitor_top_k(self, current):
        less_than_k_models = len(self.best_k_models.keys()) < self.save_top_k
        if less_than_k_models:
            return True
        return self.monitor_op(current, self.best_k_models[self.kth_best_model])

    def on_validation_end(self):
        assert self._trainer is not None, _NO_TRAINER_ERROR_MSG

        logs = self._trainer.callback_metrics
        epoch = self._trainer.current_epoch
        self.epochs_since_last_check += 1

        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}.ckpt'
            version_cnt = 0
            while os.path.isfile(filepath):
                # this epoch called before
                filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}_v{version_cnt}.ckpt'
                version_cnt += 1

            if self.save_top_k != -1:
                current = logs.get(self.monitor)

                if current is None:
                    warnings.warn(
                        f'Can save best model only with {self.monitor} available,'
                        ' skipping.', RuntimeWarning)
                else:
                    if self.check_monitor_top_k(current):

                        # remove kth
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            delpath = self.kth_best_model
                            self.best_k_models.pop(self.kth_best_model)
                            self._del_model(delpath)

                        self.best_k_models[filepath] = current
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            # monitor dict has reached k elements
                            if self.mode == 'min':
                                self.kth_best_model = max(self.best_k_models, key=self.best_k_models.get)
                            else:
                                self.kth_best_model = min(self.best_k_models, key=self.best_k_models.get)
                            self.kth_value = self.best_k_models[self.kth_best_model]

                        if self.mode == 'min':
                            self.best = min(self.best_k_models.values())
                        else:
                            self.best = max(self.best_k_models.values())
                        if self.verbose > 0:
                            log.info(
                                f'\nEpoch {epoch:05d}: {self.monitor} reached'
                                f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                                f' {filepath} as top {self.save_top_k}')
                        self._save_model(filepath)

                    else:
                        if self.verbose > 0:
                            log.info(
                                f'\nEpoch {epoch:05d}: {self.monitor}'
                                f' was not in top {self.save_top_k}')

            else:
                if self.verbose > 0:
                    log.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')
                self._save_model(filepath)


class GradientAccumulationScheduler(Callback):
    r"""
    Change gradient accumulation factor according to scheduling.

    Args:
        scheduling (dict): scheduling in format {epoch: accumulation_factor}
        .. warning:: Epochs indexing starts from "1" until v0.6.x, but will start from "0" in v0.8.0.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import GradientAccumulationScheduler

        # at epoch 5 start accumulating every 2 batches
        accumulator = GradientAccumulationScheduler(scheduling: {5: 2})
        Trainer(accumulate_grad_batches=accumulator)
    """

    def __init__(self, scheduling: dict):
        super().__init__()

        if scheduling == {}:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        for key in scheduling.keys():
            if not isinstance(key, int) or not isinstance(scheduling[key], int):
                raise TypeError("All epoches and accumulation factor must be integers")

        minimal_epoch = min(scheduling.keys())
        warnings.warn('Epochs indexing of `scheduling` starts from "1" until v0.6.x,'
                      ' but will start from "0" in v0.8.0.', DeprecationWarning)
        if minimal_epoch < 1:
            msg = f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct"
            raise IndexError(msg)
        if minimal_epoch != 1:  # if user didnt define first epoch accumulation factor
            scheduling.update({1: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_begin(self):
        assert self._trainer is not None, _NO_TRAINER_ERROR_MSG

        trainer = self._trainer
        # indexing epochs from 1 (until v0.6.x)
        # In v0.8.0, ` + 1` should be removed.
        epoch = trainer.current_epoch + 1
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break
