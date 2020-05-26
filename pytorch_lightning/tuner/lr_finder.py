from abc import ABC
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only


class TrainerLRFinderMixin(ABC):
    def _lr_finder_call_order(self):
        pass  # 
    
    def lr_find(self,
                model: LightningModule,
                train_dataloader: Optional[DataLoader] = None,
                val_dataloaders: Optional[DataLoader] = None,
                min_lr: float = 1e-8,
                max_lr: float = 1,
                num_training: int = 100,
                mode: str = 'exponential',
                early_stop_threshold: float = 4.0):
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

            early_stop_threshold: threshold for stopping the search. If the
                loss at any point is larger than early_stop_threshold*best_loss
                then the search is stopped. To disable, set to None.

        Example::

            # Setup model and trainer
            model = MyModelClass(hparams)
            trainer = pl.Trainer()

            # Run lr finder
            lr_finder = trainer.lr_find(model, ...)

            # Inspect results
            fig = lr_finder.plot(); fig.show()
            suggested_lr = lr_finder.suggestion()

            # Overwrite lr and create new model
            hparams.lr = suggested_lr
            model = MyModelClass(hparams)

            # Ready to train with new learning rate
            trainer.fit(model)

        """
        # Check for correct call order
        self._lr_finder_call_order()
        
        # Arguments we adjust during the batch size finder, save for restoring
        self.__lr_finder_dump_params(model)
        
        # Initialize lr finder callback
        lr_finder = LRFinderCallback(mode, min_lr, max_lr, num_training,
                                     early_stop_threshold)
        
        
        # Set to values that are required by the algorithm
        self.__lr_finder_reset_params(model, lr_finder, num_training)
        if self.progress_bar_callback:
            self.progress_bar_callback.disable()

        # Save initial model, that is loaded after batch size is found
        save_path = os.path.join(self.trainer.default_root_dir, 'temp_model.ckpt')
        self.trainer.save_checkpoint(str(save_path))

        
        # Configure optimizer and scheduler
        optimizers, _, _ = self.init_optimizers(model)
        if len(optimizers) != 1:
            raise MisconfigurationException(
                f'`model.configure_optimizers()` returned {len(optimizers)}, but'
                ' learning rate finder only works with single optimizer')
        model.configure_optimizers = PatchOptimizer(optimizers[0], min_lr, max_lr,
                                                    num_training, mode)

        # Fit, lr & loss logged in callback
        self.fit(model,
                 train_dataloader=train_dataloader,
                 val_dataloaders=val_dataloaders)

        # Prompt if we stopped early
        if self.global_step != num_training:
            log.info('LR finder stopped early due to diverging loss.')

        # Transfer results from callback to lr finder object
        lr_finder.results.update({'lr': self.callbacks[0].lrs,
                                  'loss': self.callbacks[0].losses})
        lr_finder._total_batch_idx = self.total_batch_idx  # for debug purpose

        # Reset model state
        self.restore(str(save_path), on_gpu=self.on_gpu)
        os.remove(save_path)

        # Finish by resetting variables so trainer is ready to fit model
        self.__lr_finder_restore_params(model)
        if self.progress_bar_callback:
            self.progress_bar_callback.enable()
        
        # Log that method was called and return object
        self._lr_find_called = True
        return lr_finder

    def __lr_finder_dump_params(self, model):
        # Prevent going into infinite loop
        self.__dumped_params = {
            'callbacks': self.trainer.callbacks,
            'logger': self.trainer.logger,
            'max_steps': self.trainer.max_steps,
            'checkpoint_callback': self.trainer.checkpoint_callback,
            'early_stop_callback': self.trainer.early_stop_callback,
            'enable_early_stop': self.trainer.enable_early_stop,
            'configure_optimizers': model.configure_optimizers,
        }
        
    def __lr_finder_reset_params(self, model, lr_finder_callback, num_training):
        self.trainer.callback = [lr_finder_callback]
        self.trainer.logger = DummyLogger()
        self.trainer.max_steps = num_training
        self.trainer.checkpoint_callback = False
        self.trainer.early_stop_callback = None
        self.trainer.enable_early_stop = False
        self.trainer.optimizers, self.trainer.schedulers = [], [],
        self.trainer.model = model
        
    def __lr_finder_restore_params(self, model):
        self.callbacks = self.__dumped_params['callbacks']
        self.logger = self.__dumped_params['logger']
        self.max_steps = self.__dumped_params['max_steps']
        self.checkpoint_callback = self.__dumped_params['checkpoint_callback']
        self.early_stop_callback = self.__dumped_params['early_stop_callback']
        self.enable_early_stop = self.__dumped_params['enable_early_stop']
        model.configure_optimizers = self.__dumped_params['configure_optimizers']
        del self.__dumped_params


class LRFinderCallback(Callback):
    """ LR finder callback. This object stores the results of Trainer.lr_find().
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
    def __init__(self, mode: str, lr_min: float, lr_max: float,
                 num_training: int, early_stop_threshold: float = 4.0,
                 beta: float = 0.98, progress_bar_refresh_rate: bool = True):
        assert mode in ('linear', 'exponential'), \
            'mode should be either `linear` or `exponential`'

        self.mode = mode
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_training = num_training
        self.early_stop_threshold = early_stop_threshold
        self.beta = beta

        self.results = {'lr': [], 'loss': []}
        self._total_batch_idx = 0  # for debug purpose

        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.progress_bar = None

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

    def suggestion(self, skip_begin: int = 10, skip_end: int = 1):
        """ This will propose a suggestion for choice of initial learning rate
        as the point with the steepest negative gradient.
        Returns:
            lr: suggested initial learning rate to use
            skip_begin: how many samples to skip in the beginning. Prevent too naive estimates
            skip_end: how many samples to skip in the end. Prevent too optimistic estimates
        """
        try:
            loss = self.results["loss"][skip_begin:-skip_end]
            min_grad = (np.gradient(np.array(loss))).argmin()
            self._optimal_idx = min_grad + skip_begin
            return self.results["lr"][self._optimal_idx]
        except Exception:
            log.exception('Failed to compute suggesting for `lr`. There might not be enough points.')
            self._optimal_idx = None

    @rank_zero_only
    def on_batch_start(self, trainer, pl_module):
        """ Called before each training batch, logs the lr that will be used """
        if (trainer.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar_refresh_rate and self.progress_bar is None:
            self.progress_bar = tqdm(desc='Finding best initial lr', total=self.num_training)

        self.results['lr'].append(trainer.lr_schedulers[0]['scheduler'].lr[0])

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        """ Called when the training batch ends, logs the calculated loss """
        if (trainer.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar:
            self.progress_bar.update()

        current_loss = trainer.running_loss.last().item()
        current_step = trainer.global_step + 1  # remove the +1 in 1.0

        # Avg loss (loss with momentum) + smoothing
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta**current_step)

        # Check if we diverging
        if self.early_stop_threshold is not None:
            if current_step > 1 and smoothed_loss > self.early_stop_threshold * self.best_loss:
                trainer.max_steps = current_step  # stop signal
                if self.progress_bar:
                    self.progress_bar.close()

        # Save best loss for diverging checking
        if smoothed_loss < self.best_loss or current_step == 1:
            self.best_loss = smoothed_loss

        self.results['loss'].append(smoothed_loss)


class PatchOptimizer(object):
    def __init__(self, optimizers, min_lr, max_lr, num_training, mode):

        new_lrs = [min_lr] * len(optimizers.param_groups)
        for param_group, new_lr in zip(optimizers.param_groups, new_lrs):
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr
        args = (optimizers, max_lr, num_training)
        self.optimizer = optimizers
        self.scheduler = _LinearLR(*args) if mode == 'linear' else _ExponentialLR(*args)

        self.patch_loader_code = str(self.__call__.__code__)

    def __call__(self):
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]


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
        return val

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
        return val
