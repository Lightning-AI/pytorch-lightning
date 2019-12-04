import os
import shutil
import logging
import warnings
import numpy as np

from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel


class Callback(object):
    """Abstract base class used to build new callbacks.

    # Properties
        * params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
            Reference of the model being trained.

    The `logs` dictionary that callback methods take as argument will contain keys
     for quantities relevant to the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class will include the following
     quantities in the `logs` that it passes to its callbacks:
    * on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
    * on_batch_begin: logs include `size`,
            the number of samples in the current batch.
    * on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).

    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        if type(model) is LightningDistributedDataParallel:
            model = model.module
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.

    """

    def __init__(self, monitor='val_loss',
                 min_delta=0.0, patience=0, verbose=0, mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            logging.info(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
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

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        stop_training = False
        if current is None:
            warnings.warn(
                f'Early stopping conditioned on metric `{self.monitor}`'
                f' which is not available. Available metrics are: {",".join(list(logs.keys()))}',
                RuntimeWarning)
            stop_training = True
            return stop_training

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop_training = True
                self.on_train_end()

        return stop_training

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            logging.info(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    The `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_top_k: if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if `save_top_k == 0`, no models are saved.
            if `save_top_k == -1`, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if `save_top_k >= 2` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode: one of {auto, min, max}.
            If `save_top_k != 0`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.

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

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
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
                            logging.info(
                                f'\nEpoch {epoch:05d}: {self.monitor} reached'
                                f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                                f' {filepath} as top {self.save_top_k}')
                        self._save_model(filepath)

                    else:
                        if self.verbose > 0:
                            logging.info(
                                f'\nEpoch {epoch:05d}: {self.monitor}'
                                f' was not in top {self.save_top_k}')

            else:
                if self.verbose > 0:
                    logging.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')
                self._save_model(filepath)


class GradientAccumulationScheduler(Callback):
    """Change gradient accumulation factor according to scheduling.

    # Arguments
        scheduling: dict, scheduling in format {epoch: accumulation_factor}

    """

    def __init__(self, scheduling: dict):
        if scheduling == {}:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        for key in scheduling.keys():
            if not isinstance(key, int) or not isinstance(scheduling[key], int):
                raise TypeError("All epoches and accumulation factor must be integers")

        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 1:
            msg = f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct"
            raise IndexError(msg)
        elif minimal_epoch != 1:  # if user didnt define first epoch accumulation factor
            scheduling.update({1: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_begin(self, epoch, trainer):
        epoch += 1  # indexing epochs from 1
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break


# if __name__ == '__main__':
#     c = EarlyStopping(min_delta=0.9, patience=2, verbose=True)
#     losses = [10, 9, 8, 8, 6, 4.3, 5, 4.4, 2.8, 2.5]
#     for i, loss in enumerate(losses):
#         should_stop = c.on_epoch_end(i, logs={'val_loss': loss})
#         logging.info(loss)
#         if should_stop:
#             break
