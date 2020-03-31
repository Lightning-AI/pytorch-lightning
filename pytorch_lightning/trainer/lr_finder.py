from typing import Optional
from abc import ABC
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class TrainerLRFinderMixin(ABC):
    def find_lr(self, 
                model: LightningModule,
                train_dataloader: Optional[DataLoader] = None,
                min_lr: float = 1e-6, 
                max_lr: float = 1,
                num_iters: int = 100,
                mode: str = 'exponential',
                num_accumulation_steps = 10): 
        
        lr_finder = _LRFinder(mode, min_lr, max_lr, num_iters, num_accumulation_steps)
        
        # Use special lr logger callback
        callbacks = self.callbacks
        self.callbacks = [LRCallback(num_iters, 
                                     num_accumulation_steps,
                                     show_progress_bar=True)]
        
        # No logging
        logger = self.logger
        self.logger = None
        
        # Max step set to number of iterations
        max_steps = self.max_steps
        self.max_steps = num_iters
        
        # Progress bar does not make much sense
        show_progress_bar = self.show_progress_bar
        self.show_progress_bar = False
        
        optimizer, _ = self.init_optimizers(model.configure_optimizers())
        assert len(optimizer)==1, 'cannot find lr for more than 1 optimizer'
        old_configure_optimizers = model.configure_optimizers
        model.configure_optimizers = lr_finder._get_new_optimizer(optimizer[0])
        
        # Fit and log lr/loss
        self.fit(model)
        
        lr_finder.history.update({'lr': self.callbacks[0].lrs,
                                  'loss': self.callbacks[0].losses})
        
        # Finish by resetting variables
        self.logger = logger
        self.callbacks = callbacks
        self.max_steps = max_steps
        self.show_progress_bar = show_progress_bar
        model.configure_optimizers = old_configure_optimizers
        
        return lr_finder
        
class _LRFinder(object):
    def __init__(self, mode, lr_min, lr_max, num_iters, num_accumulation_steps):
        assert mode in ('linear', 'exponential'), \
            'mode should be either `linear` or `exponential`'
        
        self.mode = mode
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_iters = num_iters
        self.num_accumulation_steps = num_accumulation_steps
        
        self.history = {}
        
    def _get_new_optimizer(self, optimizer):
        new_lrs = [self.lr_min] * len(optimizer.param_groups)
        for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr
        
        args = (optimizer, self.lr_max, self.num_iters)
        scheduler = LinearLR(*args) if self.mode == 'linear' else ExponentialLR(*args)
        
        def configure_optimizers():
            return [optimizer], [{'scheduler': scheduler, 
                                  'interval': 'step',
                                  'frequency': self.num_accumulation_steps}]
        
        return configure_optimizers
    
    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        
        lrs = self.history["lr"]
        losses = self.history["loss"]
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if self.mode == 'exponential':
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        
        if fig is not None:
            plt.show()

        return ax


class LRCallback(Callback):
    def __init__(self, num_iters, num_accumulation_steps, show_progress_bar=False):
        self.num_iters = num_iters
        self.num_accumulation_steps = num_accumulation_steps
        self.losses = [ ]
        self.lrs = [ ]
        if show_progress_bar:
            self.progress_bar = tqdm(desc='Finding best initial lr', 
                                     total=num_iters*num_accumulation_steps)
        else:
            self.progress_bar = None
            
    def on_batch_start(self, trainer, pl_module):
        """ Called before each training batch, logs the lr that will be used """
        self.lrs.append(trainer.lr_schedulers[0]['scheduler'].get_last_lr())
        
    def on_batch_end(self, trainer, pl_module):
        """ Called when the training batch ends, logs the calculated loss """
        self.losses.append(np.mean(trainer.running_loss[-self.num_accumulation_steps:]))
        if self.progress_bar:
            self.progress_bar.update()
        
class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        if self.last_epoch > 0:
            return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]    

class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        
        if self.last_epoch > 0:
            return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]