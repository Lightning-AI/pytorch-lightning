"""
Trainer Learning Rate Finder
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainerLRFinderMixin(ABC):
    @abstractmethod
    def save_checkpoint(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def restore(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def _run_lr_finder_internally(self, model: LightningModule):
        """ Call lr finder internally during Trainer.fit() """
        lr_finder = self.lr_find(model)
        lr = lr_finder.suggestion()
        # TODO: log lr.results to self.logger
        if isinstance(self.auto_lr_find, str):
            # Try to find requested field, may be nested
            if _nested_hasattr(model.hparams, self.auto_lr_find):
                _nested_setattr(model.hparams, self.auto_lr_find, lr)
            else:
                raise MisconfigurationException(
                    f'`auto_lr_find` was set to {self.auto_lr_find}, however'
                    ' could not find this as a field in `model.hparams`.')
        else:
            if hasattr(model.hparams, 'lr'):
                model.hparams.lr = lr
            elif hasattr(model.hparams, 'learning_rate'):
                model.hparams.learning_rate = lr
            else:
                raise MisconfigurationException(
                    'When auto_lr_find is set to True, expects that hparams'
                    ' either has field `lr` or `learning_rate` that can overridden')
        log.info(f'Learning rate set to {lr}')

    def lr_find(self,
                model: LightningModule,
                train_dataloader: Optional[DataLoader] = None,
                min_lr: float = 1e-8,
                max_lr: float = 1,
                num_training: int = 100,
                mode: str = 'exponential',
                num_accumulation_steps: int = 1):
        r"""
        lr_find enables the user to do a range test of good initial learning rates,
        to reduce the amount of guesswork in picking a good starting learning rate.

        Args:
            model: Model to do range testing for

            train_dataloader: A PyTorch
                DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            min_lr: minimum learning rate to investigate

            max_lr: maximum learning rate to investigate

            num_training: number of learning rates to test

            mode: search strategy, either 'linear' or 'exponential'. If set to
                'linear' the learning rate will be searched by linearly increasing
                after each batch. If set to 'exponential', will increase learning
                rate exponentially.

            num_accumulation_steps: number of batches to calculate loss over.

        Example::

            # Setup model and trainer
            model = MyModelClass(hparams)
            trainer = pl.Trainer()

            # Run lr finder
            lr_finder = trainer.lr_find(model, ...)

            # Inspect results
            fig = lr_finder.plot(); fig.show()
            suggested_lr = lr_finder.suggest()

            # Overwrite lr and create new model
            hparams.lr = suggested_lr
            model = MyModelClass(hparams)

            # Ready to train with new learning rate
            trainer.fit(model)

        """
        save_path = os.path.join(self.default_root_dir, 'lr_find_temp.ckpt')

        self._dump_params(model)

        # Prevent going into infinite loop
        self.auto_lr_find = False

        # Initialize lr finder object (stores results)
        lr_finder = _LRFinder(mode, min_lr, max_lr, num_training)

        # Use special lr logger callback
        self.callbacks = [_LRCallback(num_training, show_progress_bar=True)]

        # No logging
        self.logger = None

        # Max step set to number of iterations
        self.max_steps = num_training

        # Disable standard progress bar for fit
        self.progress_bar_refresh_rate = False

        # Accumulation of gradients
        self.accumulate_grad_batches = num_accumulation_steps

        # Disable standard checkpoint
        self.checkpoint_callback = False

        # Required for saving the model
        self.optimizers, self.schedulers = [], [],
        self.model = model

        # Dump model checkpoint
        self.save_checkpoint(str(save_path))

        # Configure optimizer and scheduler
        optimizers, _, _ = self.init_optimizers(model)

        if len(optimizers) != 1:
            raise MisconfigurationException(
                f'`model.configure_optimizers()` returned {len(optimizers)}, but'
                ' learning rate finder only works with single optimizer')
        configure_optimizers = model.configure_optimizers
        model.configure_optimizers = lr_finder._get_new_optimizer(optimizers[0])

        # Fit, lr & loss logged in callback
        self.fit(model, train_dataloader=train_dataloader)

        # Prompt if we stopped early
        if self.global_step != num_training:
            log.info('LR finder stopped early due to diverging loss.')

        # Transfer results from callback to lr finder object
        lr_finder.results.update({'lr': self.callbacks[0].lrs,
                                  'loss': self.callbacks[0].losses})

        # Reset model state
        self.restore(str(save_path), on_gpu=self.on_gpu)
        os.remove(save_path)

        # Finish by resetting variables so trainer is ready to fit model
        self._restore_params(model)

        return lr_finder

    def _dump_params(self, model):
        # Prevent going into infinite loop
        self._params = {
            'auto_lr_find': self.auto_lr_find,
            'callbacks': self.callbacks,
            'logger': self.logger,
            'max_steps': self.max_steps,
            'progress_bar_refresh_rate': self.progress_bar_refresh_rate,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'checkpoint_callback': self.checkpoint_callback,
            'configure_optimizers': model.configure_optimizers,
        }

    def _restore_params(self, model):
        self.auto_lr_find = self._params['auto_lr_find']
        self.logger = self._params['logger']
        self.callbacks = self._params['callbacks']
        self.max_steps = self._params['max_steps']
        self.progress_bar_refresh_rate = self._params['progress_bar_refresh_rate']
        self.accumulate_grad_batches = self._params['accumulate_grad_batches']
        self.checkpoint_callback = self._params['checkpoint_callback']
        model.configure_optimizers = self._params['configure_optimizers']


class _LRFinder(object):
    """ LR finder object. This object stores the results of Trainer.lr_find().

    Args:
        mode: either `linear` or `exponential`, how to increase lr after each step

        lr_min: lr to start search from

        lr_max: lr to stop seach

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

    def _get_new_optimizer(self, optimizer: torch.optim.Optimizer):
        """ Construct a new `configure_optimizers()` method, that has a optimizer
            with initial lr set to lr_min and a scheduler that will either
            linearly or exponentially increase the lr to lr_max in num_training steps.

        Args:
            optimizer: instance of `torch.optim.Optimizer`

        """
        new_lrs = [self.lr_min] * len(optimizer.param_groups)
        for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr

        args = (optimizer, self.lr_max, self.num_training)
        scheduler = _LinearLR(*args) if self.mode == 'linear' else _ExponentialLR(*args)

        def configure_optimizers():
            return [optimizer], [{'scheduler': scheduler,
                                  'interval': 'step'}]

        return configure_optimizers

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
                ax.plot(lrs[self._optimal_idx], losses[self._optimal_idx],
                        markersize=10, marker='o', color='red')

        if show:
            plt.show()

        return fig

    def suggestion(self):
        """ This will propose a suggestion for choice of initial learning rate
        as the point with the steepest negative gradient.

        Returns:
            lr: suggested initial learning rate to use

        """
        try:
            min_grad = (np.gradient(np.array(self.results["loss"]))).argmin()
            self._optimal_idx = min_grad
            return self.results["lr"][min_grad]
        except Exception:
            log.warning('Failed to compute suggesting for `lr`.'
                        ' There might not be enough points.')
            self._optimal_idx = None


class _LRCallback(Callback):
    """ Special callback used by the learning rate finder. This callbacks log
    the learning rate before each batch and log the corresponding loss after
    each batch. """
    def __init__(self, num_training: int, show_progress_bar: bool = False, beta: float = 0.98):
        self.num_training = num_training
        self.beta = beta
        self.losses = []
        self.lrs = []
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.show_progress_bar = show_progress_bar
        self.progress_bar = None

    def on_batch_start(self, trainer, pl_module):
        """ Called before each training batch, logs the lr that will be used """
        if self.show_progress_bar and self.progress_bar is None:
            self.progress_bar = tqdm(desc='Finding best initial lr', total=self.num_training)

        self.lrs.append(trainer.lr_schedulers[0]['scheduler'].lr[0])

    def on_batch_end(self, trainer, pl_module):
        """ Called when the training batch ends, logs the calculated loss """
        if self.progress_bar:
            self.progress_bar.update()

        current_loss = trainer.running_loss.last().item()
        current_step = trainer.global_step + 1  # remove the +1 in 1.0

        # Avg loss (loss with momentum) + smoothing
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta**current_step)

        # Check if we diverging
        if current_step > 1 and smoothed_loss > 4 * self.best_loss:
            trainer.max_steps = current_step  # stop signal
            if self.progress_bar:
                self.progress_bar.close()

        # Save best loss for diverging checking
        if smoothed_loss < self.best_loss or current_step == 1:
            self.best_loss = smoothed_loss

        self.losses.append(smoothed_loss)


class _LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries
    over a number of iterations.
    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 end_lr: float,
                 num_iter: int,
                 last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
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

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 end_lr: float,
                 num_iter: int,
                 last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
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


def _nested_hasattr(obj, path):
    parts = path.split(".")
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return False
    else:
        return True


def _nested_setattr(obj, path, val):
    parts = path.split(".")
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
    setattr(obj, parts[-1], val)
