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
import importlib
import logging
import os
from functools import wraps
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities import DeviceType, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

log = logging.getLogger(__name__)


def _determine_lr_attr_name(trainer: 'pl.Trainer', model: 'pl.LightningModule') -> str:
    if isinstance(trainer.auto_lr_find, str):
        if not lightning_hasattr(model, trainer.auto_lr_find):
            raise MisconfigurationException(
                f'`auto_lr_find` was set to {trainer.auto_lr_find}, however'
                ' could not find this as a field in `model` or `model.hparams`.'
            )
        return trainer.auto_lr_find

    attr_options = ('lr', 'learning_rate')
    for attr in attr_options:
        if lightning_hasattr(model, attr):
            return attr

    raise MisconfigurationException(
        'When `auto_lr_find=True`, either `model` or `model.hparams` should'
        f' have one of these fields: {attr_options} overridden.'
    )


class _LRFinder(object):
    """ LR finder object. This object stores the results of lr_find().

    Args:
        mode: either `linear` or `exponential`, how to increase lr after each step

        lr_min: lr to start search from

        lr_max: lr to stop search

        num_training: number of steps to take between lr_min and lr_max

    Example::
        # Run lr finder
        lr_finder = trainer.lr_find(model)

        # Results stored in
        lr_finder.results

        # Plot using
        lr_finder.plot()

        # Get suggestion
        lr = lr_finder.suggestion()
    """

    def __init__(self, mode: str, lr_min: float, lr_max: float, num_training: int):
        assert mode in ('linear', 'exponential'), \
            'mode should be either `linear` or `exponential`'

        self.mode = mode
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_training = num_training

        self.results = {}
        self._total_batch_idx = 0  # for debug purpose

    def _exchange_scheduler(self, configure_optimizers: Callable):
        """ Decorate configure_optimizers methods such that it returns the users
            originally specified optimizer together with a new scheduler that
            that takes care of the learning rate search.
        """

        @wraps(configure_optimizers)
        def func():
            # Decide the structure of the output from configure_optimizers
            # Same logic as method `init_optimizers` in trainer/optimizers.py
            optim_conf = configure_optimizers()
            if isinstance(optim_conf, Optimizer):
                optimizers = [optim_conf]
            elif isinstance(optim_conf, (list, tuple)) and len(optim_conf) == 2 \
                    and isinstance(optim_conf[0], list):
                optimizers, _ = optim_conf
            elif isinstance(optim_conf, dict):
                optimizers = [optim_conf["optimizer"]]
            elif isinstance(optim_conf, (list, tuple)) and isinstance(optim_conf[0], dict):
                optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
            elif isinstance(optim_conf, (list, tuple)):
                optimizers = [optim_conf]

            if len(optimizers) != 1:
                raise MisconfigurationException(
                    f'`model.configure_optimizers()` returned {len(optimizers)}, but'
                    ' learning rate finder only works with single optimizer'
                )

            optimizer = optimizers[0]

            new_lrs = [self.lr_min] * len(optimizer.param_groups)
            for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
                param_group["lr"] = new_lr
                param_group["initial_lr"] = new_lr

            args = (optimizer, self.lr_max, self.num_training)
            scheduler = _LinearLR(*args) if self.mode == 'linear' else _ExponentialLR(*args)

            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

        return func

    def plot(self, suggest: bool = False, show: bool = False):
        """ Plot results from lr_find run
        Args:
            suggest: if True, will mark suggested lr to use with a red point

            show: if True, will show figure
        """
        import matplotlib.pyplot as plt

        lrs = self.results["lr"]
        losses = self.results["loss"]

        fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if self.mode == 'exponential':
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if suggest:
            _ = self.suggestion()
            if self._optimal_idx:
                ax.plot(lrs[self._optimal_idx], losses[self._optimal_idx], markersize=10, marker='o', color='red')

        if show:
            plt.show()

        return fig

    def suggestion(self, skip_begin: int = 10, skip_end: int = 1):
        """ This will propose a suggestion for choice of initial learning rate
        as the point with the steepest negative gradient.

        Returns:
            lr: suggested initial learning rate to use
            skip_begin: how many samples to skip in the beginning. Prevent too naive estimates
            skip_end: how many samples to skip in the end. Prevent too optimistic estimates

        """
        try:
            loss = np.array(self.results["loss"][skip_begin:-skip_end])
            loss = loss[np.isfinite(loss)]
            min_grad = np.gradient(loss).argmin()
            self._optimal_idx = min_grad + skip_begin
            return self.results["lr"][self._optimal_idx]
        # todo: specify the possible exception
        except Exception:
            log.exception('Failed to compute suggesting for `lr`. There might not be enough points.')
            self._optimal_idx = None


def lr_find(
        trainer: 'pl.Trainer',
        model: 'pl.LightningModule',
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = 'exponential',
        early_stop_threshold: float = 4.0,
        update_attr: bool = False,
) -> Optional[_LRFinder]:
    """See :meth:`~pytorch_lightning.tuner.tuning.Tuner.lr_find`"""
    if trainer.fast_dev_run:
        rank_zero_warn('Skipping learning rate finder since fast_dev_run is enabled.', UserWarning)
        return

    # Determine lr attr
    if update_attr:
        lr_attr_name = _determine_lr_attr_name(trainer, model)

    save_path = os.path.join(trainer.default_root_dir, 'lr_find_temp_model.ckpt')

    __lr_finder_dump_params(trainer, model)

    # Prevent going into infinite loop
    trainer.auto_lr_find = False

    # Initialize lr finder object (stores results)
    lr_finder = _LRFinder(mode, min_lr, max_lr, num_training)

    # Use special lr logger callback
    trainer.callbacks = [_LRCallback(num_training, early_stop_threshold, progress_bar_refresh_rate=1)]

    # No logging
    trainer.logger = DummyLogger()

    # Max step set to number of iterations
    trainer.max_steps = num_training

    # Disable standard progress bar for fit
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    # Required for saving the model
    trainer.optimizers, trainer.schedulers = [], [],
    trainer.model = model

    # Dump model checkpoint
    trainer.save_checkpoint(str(save_path))

    # Configure optimizer and scheduler
    model.configure_optimizers = lr_finder._exchange_scheduler(model.configure_optimizers)

    # Fit, lr & loss logged in callback
    trainer.tuner._run(model)

    # Prompt if we stopped early
    if trainer.global_step != num_training:
        log.info(f'LR finder stopped early after {trainer.global_step} steps due to diverging loss.')

    # Transfer results from callback to lr finder object
    lr_finder.results.update({'lr': trainer.callbacks[0].lrs, 'loss': trainer.callbacks[0].losses})
    lr_finder._total_batch_idx = trainer.total_batch_idx  # for debug purpose

    # Reset model state
    if trainer.is_global_zero:
        trainer.checkpoint_connector.restore(str(save_path), on_gpu=trainer._device_type == DeviceType.GPU)
        fs = get_filesystem(str(save_path))
        if fs.exists(save_path):
            fs.rm(save_path)

    # Finish by resetting variables so trainer is ready to fit model
    __lr_finder_restore_params(trainer, model)
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    # Update lr attr if required
    if update_attr:
        lr = lr_finder.suggestion()

        # TODO: log lr.results to self.logger
        lightning_setattr(model, lr_attr_name, lr)
        log.info(f'Learning rate set to {lr}')

    return lr_finder


def __lr_finder_dump_params(trainer, model):
    # Prevent going into infinite loop
    trainer.__dumped_params = {
        'auto_lr_find': trainer.auto_lr_find,
        'callbacks': trainer.callbacks,
        'logger': trainer.logger,
        'max_steps': trainer.max_steps,
        'checkpoint_callback': trainer.checkpoint_callback,
        'configure_optimizers': model.configure_optimizers,
    }


def __lr_finder_restore_params(trainer, model):
    trainer.auto_lr_find = trainer.__dumped_params['auto_lr_find']
    trainer.logger = trainer.__dumped_params['logger']
    trainer.callbacks = trainer.__dumped_params['callbacks']
    trainer.max_steps = trainer.__dumped_params['max_steps']
    model.configure_optimizers = trainer.__dumped_params['configure_optimizers']
    del trainer.__dumped_params


class _LRCallback(Callback):
    """ Special callback used by the learning rate finder. This callbacks log
    the learning rate before each batch and log the corresponding loss after
    each batch.

    Args:
        num_training: number of iterations done by the learning rate finder
        early_stop_threshold: threshold for stopping the search. If the
            loss at any point is larger than ``early_stop_threshold*best_loss``
            then the search is stopped. To disable, set to ``None``.
        progress_bar_refresh_rate: rate to refresh the progress bar for
            the learning rate finder
        beta: smoothing value, the loss being logged is a running average of
            loss values logged until now. ``beta`` controls the forget rate i.e.
            if ``beta=0`` all past information is ignored.

    """

    def __init__(
            self,
            num_training: int,
            early_stop_threshold: float = 4.0,
            progress_bar_refresh_rate: int = 0,
            beta: float = 0.98
    ):
        self.num_training = num_training
        self.early_stop_threshold = early_stop_threshold
        self.beta = beta
        self.losses = []
        self.lrs = []
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.progress_bar = None

    def on_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """ Called before each training batch, logs the lr that will be used """
        if (trainer.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar_refresh_rate and self.progress_bar is None:
            self.progress_bar = tqdm(desc='Finding best initial lr', total=self.num_training)

        self.lrs.append(trainer.lr_schedulers[0]['scheduler'].lr[0])

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch,
                           batch_idx: Optional[int],
                           dataloader_idx: Optional[int]) -> None:
        """ Called when the training batch ends, logs the calculated loss """
        if (trainer.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar:
            self.progress_bar.update()

        current_loss = trainer.train_loop.running_loss.last().item()
        current_step = trainer.global_step

        # Avg loss (loss with momentum) + smoothing
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (current_step + 1))

        # Check if we diverging
        if self.early_stop_threshold is not None:
            if current_step > 1 and smoothed_loss > self.early_stop_threshold * self.best_loss:
                trainer.max_steps = current_step  # stop signal
                if self.progress_bar:
                    self.progress_bar.close()

        # Save best loss for diverging checking
        if smoothed_loss < self.best_loss or current_step == 1:
            self.best_loss = smoothed_loss

        self.losses.append(smoothed_loss)


class _LinearLR(_LRScheduler):
    """
    Linearly increases the learning rate between two boundaries over a number of iterations.

    Args:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """
    last_epoch: int
    base_lrs: Sequence

    def __init__(self, optimizer: torch.optim.Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self):
        return self._lr


class _ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries
    over a number of iterations.

    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """
    last_epoch: int
    base_lrs: Sequence

    def __init__(self, optimizer: torch.optim.Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self):
        return self._lr
