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

"""
Model Checkpointing
===================

Automatically save model checkpoints during training.

"""

import os
import re
from typing import Optional

import numpy as np
import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning.utilities.cloud_io import gfile, makedirs, is_remote_path


class ModelCheckpoint(Callback):
    r"""
    Save the model after every epoch if it improves.

    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to retrieve its score.

    Args:
        filepath: path to save the model file.
            Can contain named formatting options to be auto-filled.

            Example::

                # custom path
                # saves a file like: my/path/epoch_0.ckpt
                >>> checkpoint_callback = ModelCheckpoint('my/path/')

                # save any arbitrary metrics like `val_loss`, etc. in name
                # saves a file like: my/path/epoch=2-val_loss=0.2_other_metric=0.3.ckpt
                >>> checkpoint_callback = ModelCheckpoint(
                ...     filepath='my/path/{epoch}-{val_loss:.2f}-{other_metric:.2f}'
                ... )

            By default, filepath is `None` and will be set at runtime to the location
            specified by :class:`~pytorch_lightning.trainer.trainer.Trainer`'s
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.default_root_dir` or
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_save_path` arguments,
            and if the Trainer uses a logger, the path will also contain logger name and version.

        monitor: quantity to monitor.
        verbose: verbosity mode. Default: ``False``.
        save_last: always saves the model at the end of the epoch. Default: ``False``.
        save_top_k: if ``save_top_k == k``,
            the best k models according to
            the quantity monitored will be saved.
            if ``save_top_k == 0``, no models are saved.
            if ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if ``save_top_k >= 2`` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode: one of {auto, min, max}.
            If ``save_top_k != 0``, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if ``True``, then only the model's weights will be
            saved (``model.save_weights(filepath)``), else the full model
            is saved (``model.save(filepath)``).
        period: Interval (number of epochs) between checkpoints.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import ModelCheckpoint

        # saves checkpoints to 'my/path/' whenever 'val_loss' has a new min
        >>> checkpoint_callback = ModelCheckpoint(filepath='my/path/')
        >>> trainer = Trainer(checkpoint_callback=checkpoint_callback)

        # save epoch and val_loss in name
        # saves a file like: my/path/sample-mnist_epoch=02_val_loss=0.32.ckpt
        >>> checkpoint_callback = ModelCheckpoint(
        ...     filepath='my/path/sample-mnist_{epoch:02d}-{val_loss:.2f}'
        ... )

        # retrieve the best checkpoint after training
        checkpoint_callback = ModelCheckpoint(filepath='my/path/')
        trainer = Trainer(checkpoint_callback=checkpoint_callback)
        model = ...
        trainer.fit(model)
        checkpoint_callback.best_model_path

    """

    CHECKPOINT_NAME_LAST = "last.ckpt"
    CHECKPOINT_STATE_BEST_SCORE = "checkpoint_callback_best_model_score"
    CHECKPOINT_STATE_BEST_PATH = "checkpoint_callback_best_model_path"

    def __init__(self, filepath: Optional[str] = None, monitor: str = 'val_loss', verbose: bool = False,
                 save_last: bool = False, save_top_k: int = 1, save_weights_only: bool = False,
                 mode: str = 'auto', period: int = 1, prefix: str = ''):
        super().__init__()
        if(filepath):
            filepath = str(filepath)  # the tests pass in a py.path.local but we want a str
        if save_top_k > 0 and filepath is not None and gfile.isdir(filepath) and len(gfile.listdir(filepath)) > 0:
            rank_zero_warn(
                f"Checkpoint directory {filepath} exists and is not empty with save_top_k != 0."
                "All files in this directory will be deleted when a checkpoint is saved!"
            )
        self._rank = 0

        self.monitor = monitor
        self.verbose = verbose
        if filepath is None:  # will be determined by trainer at runtime
            self.dirpath, self.filename = None, None
        else:
            if gfile.isdir(filepath):
                self.dirpath, self.filename = filepath, '{epoch}'
            else:
                if not is_remote_path(filepath):  # dont normalize remote paths
                    filepath = os.path.realpath(filepath)
                self.dirpath, self.filename = os.path.split(filepath)
            makedirs(self.dirpath)  # calls with exist_ok
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.period = period
        self.epoch_last_check = None
        self.prefix = prefix
        self.best_k_models = {}
        # {filename: monitor}
        self.kth_best_model_path = ''
        self.best_model_score = 0
        self.best_model_path = ''
        self.save_function = None
        self.warned_result_obj = False

        torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max'),
            'auto': (-torch_inf, 'max') if 'acc' in self.monitor or self.monitor.startswith('fmeasure')
            else (torch_inf, 'min'),
        }

        if mode not in mode_dict:
            rank_zero_warn(f'ModelCheckpoint mode {mode} is unknown, '
                           f'fallback to auto mode.', RuntimeWarning)
            mode = 'auto'

        self.kth_value, self.mode = mode_dict[mode]

    @property
    def best(self):
        rank_zero_warn("Attribute `best` has been renamed to `best_model_score` since v0.8.0"
                       " and will be removed in v0.10.0", DeprecationWarning)
        return self.best_model_score

    @property
    def kth_best_model(self):
        rank_zero_warn("Attribute `kth_best_model` has been renamed to `kth_best_model_path` since v0.8.0"
                       " and will be removed in v0.10.0", DeprecationWarning)
        return self.kth_best_model_path

    def _del_model(self, filepath):
        if gfile.exists(filepath):
            try:
                # in compat mode, remove is not implemented so if running this
                # against an actual remove file system and the correct remote
                # dependencies exist then this will work fine.
                gfile.remove(filepath)
            except AttributeError:
                if is_remote_path(filepath):
                    log.warning("Unable to remove stale checkpoints due to running gfile in compatibility mode."
                                " Please install tensorflow to run gfile in full mode"
                                " if writing checkpoints to remote locations")
                else:
                    os.remove(filepath)

    def _save_model(self, filepath, trainer, pl_module):

        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # make paths
        if not gfile.exists(os.path.dirname(filepath)):
            makedirs(os.path.dirname(filepath))

        # delegate the saving to the model
        if self.save_function is not None:
            self.save_function(filepath, self.save_weights_only)
        else:
            raise ValueError(".save_function() not set")

    def check_monitor_top_k(self, current):
        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        if not isinstance(current, torch.Tensor):
            rank_zero_warn(
                f'{current} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly.'
                f' HINT: check the value of {self.monitor} in your validation loop', RuntimeWarning
            )
            current = torch.tensor(current)

        monitor_op = {
            "min": torch.lt,
            "max": torch.gt,
        }[self.mode]

        return monitor_op(current, self.best_k_models[self.kth_best_model_path])

    def format_checkpoint_name(self, epoch, metrics, ver=None):
        """Generate a filename according to the defined template.

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(0, {}))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch:03d}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(5, {}))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}-{val_loss:.2f}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(2, dict(val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{missing:d}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(0, {}))
            'missing=0.ckpt'
        """
        # check if user passed in keys to the string
        groups = re.findall(r'(\{.*?)[:\}]', self.filename)

        if len(groups) == 0:
            # default name
            filename = f'{self.prefix}_ckpt_epoch_{epoch}'
        else:
            metrics['epoch'] = epoch
            filename = self.filename
            for tmp in groups:
                name = tmp[1:]
                filename = filename.replace(tmp, name + '={' + name)
                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)
        str_ver = f'_v{ver}' if ver is not None else ''
        filepath = os.path.join(self.dirpath, self.prefix + filename + str_ver + '.ckpt')
        return filepath

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:

        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path

        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        if self.dirpath is not None:
            return  # short circuit

        self.filename = '{epoch}'

        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # the user has changed weights_save_path, it overrides anything
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = trainer.logger.version if isinstance(
                trainer.logger.version, str) else f'version_{trainer.logger.version}'
            ckpt_path = os.path.join(
                save_dir,
                trainer.logger.name,
                version,
                "checkpoints"
            )
        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        self.dirpath = ckpt_path

        assert trainer.global_rank == 0, 'tried to make a checkpoint from non global_rank=0'
        if not gfile.exists(self.dirpath):
            makedirs(self.dirpath)

    def __warn_deprecated_monitor_key(self):
        using_result_obj = os.environ.get('PL_USING_RESULT_OBJ', None)
        invalid_key = self.monitor not in ['val_loss', 'checkpoint_on', 'loss', 'val_checkpoint_on']
        if using_result_obj and not self.warned_result_obj and invalid_key:
            self.warned_result_obj = True
            m = f"""
                    When using EvalResult(early_stop_on=X) or TrainResult(early_stop_on=X) the
                    'monitor' key of ModelCheckpoint has no effect.
                    Remove ModelCheckpoint(monitor='{self.monitor}) to fix')
                """
            rank_zero_warn(m)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # only run on main process
        if trainer.global_rank != 0:
            return

        # TODO: remove when dict results are deprecated
        self.__warn_deprecated_monitor_key()

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # support structured results
        if metrics.get('checkpoint_on') is not None:
            self.monitor = 'checkpoint_on'

        # conditioned val metrics override conditioned train loop metrics
        if metrics.get('val_checkpoint_on') is not None:
            self.monitor = 'val_checkpoint_on'

        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epoch_last_check is not None and (epoch - self.epoch_last_check) < self.period:
            # skipping in this term
            return

        self.epoch_last_check = epoch

        ckpt_name_metrics = trainer.logged_metrics
        filepath = self.format_checkpoint_name(epoch, ckpt_name_metrics)
        version_cnt = 0
        while gfile.exists(filepath):
            filepath = self.format_checkpoint_name(epoch, ckpt_name_metrics, ver=version_cnt)
            # this epoch called before
            version_cnt += 1

        if self.save_top_k != -1:
            current = metrics.get(self.monitor)

            if not isinstance(current, torch.Tensor):
                rank_zero_warn(
                    f'The metric you returned {current} must be a `torch.Tensor` instance, checkpoint not saved'
                    f' HINT: what is the value of {self.monitor} in validation_epoch_end()?', RuntimeWarning
                )
                if current is not None:
                    current = torch.tensor(current)

            if current is None:
                rank_zero_warn(
                    f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning
                )
            elif self.check_monitor_top_k(current):
                self._do_check_save(filepath, current, epoch, trainer, pl_module)
            elif self.verbose > 0:
                log.info(f'\nEpoch {epoch:05d}: {self.monitor}  was not in top {self.save_top_k}')

        else:
            if self.verbose > 0:
                log.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')

            assert trainer.global_rank == 0, 'tried to make a checkpoint from non global_rank=0'
            self._save_model(filepath, trainer, pl_module)

        if self.save_last:
            filepath = os.path.join(self.dirpath, self.prefix + ModelCheckpoint.CHECKPOINT_NAME_LAST)
            self._save_model(filepath, trainer, pl_module)

    def _do_check_save(self, filepath, current, epoch, trainer, pl_module):
        # remove kth

        del_list = []
        if len(self.best_k_models) == self.save_top_k and self.save_top_k > 0:
            delpath = self.kth_best_model_path
            self.best_k_models.pop(self.kth_best_model_path)
            del_list.append(delpath)

        self.best_k_models[filepath] = current
        if len(self.best_k_models) == self.save_top_k:
            # monitor dict has reached k elements
            _op = max if self.mode == 'min' else min
            self.kth_best_model_path = _op(self.best_k_models,
                                           key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == 'min' else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose > 0:
            log.info(
                f'\nEpoch {epoch:05d}: {self.monitor} reached'
                f' {current:0.5f} (best {self.best_model_score:0.5f}), saving model to'
                f' {filepath} as top {self.save_top_k}')
        self._save_model(filepath, trainer, pl_module)

        for cur_path in del_list:
            if cur_path != filepath:
                self._del_model(cur_path)
