from abc import ABC

import math
import sys
from abc import ABC, abstractmethod
import gc
import os
from typing import Optional
import time

import numpy as np
import torch
from torch import Tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import is_oom_error, garbage_collection_cuda
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr, lightning_getattr

class TunerBatchScalerMixin(ABC):
    def _batch_scaler_call_order(self):
        if self._lr_find_called:
            rank_zero_warn(
                'You called learning rate finder before batch scaler. Please note'
                ' that the result of the learning rate finder is influenced by the'
                ' batch size, and the batch size should therefore be called before'
                ' the learning rate finder',
                UserWarning)
    
    def scale_batch_size(self,
                         model: LightningModule,
                         mode: str = 'power',
                         steps_per_trial: int = 3,
                         init_val: int = 2,
                         max_trials: int = 15,
                         attribute_name: str = 'batch_size'):
        r"""
        Will iteratively try to find the largest batch size for a given model
        that does not give an out of memory (OOM) error.

        Args:
            model: Model to fit.

            mode: string setting the search mode. Either `power` or `binsearch`.
                If mode is `power` we keep multiplying the batch size by 2, until
                we get an OOM error. If mode is 'binsearch', we will initially
                also keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the
                batch size thTrainerLRFinderMixinat failed.

            steps_per_trial: number of steps to run with a given batch size.
                Idealy 1 should be enough to test if a OOM error occurs,
                however in practise a few are needed

            init_val: initial batch size to start the search with

            max_trials: max number of increase in batch size done before
               algorithm is terminated
               
            attribute_name: name of field that changes the batch_size of model

        """
        # Check for correct call order
        self._batch_scaler_call_order()
        
        if not lightning_hasattr(model, attribute_name):
            raise MisconfigurationException(f'Field {attribute_name} not found in `model` namespace')

        if hasattr(model.train_dataloader, 'patch_loader_code'):
            raise MisconfigurationException('The batch scaling feature cannot be used with dataloaders'
                                            ' passed directly to `.fit()`. Please disable the feature or'
                                            ' incorporate the dataloader into the model.')

        # Arguments we adjust during the batch size finder, save for restoring
        self.__scale_batch_dump_params()

        # Set to values that are required by the algorithm
        self.__scale_batch_reset_params(model, steps_per_trial)
        if self.trainer.progress_bar_callback:
            self.trainer.progress_bar_callback.disable()
            
        # Save initial model, that is loaded after batch size is found
        save_path = os.path.join(self.trainer.default_root_dir, 'temp_model.ckpt')
        self.trainer.save_checkpoint(str(save_path))

        # Initially we just double in size until an OOM is encountered
        new_size = _adjust_batch_size(self.trainer, value=init_val)  # initially set to init_val
        if mode == 'power':
            batch_scaler = _run_power_scaling(self.trainer, model, new_size, attribute_name, max_trials)
        elif mode == 'binsearch':
            batch_scaler = _run_binsearch_scaling(self.trainer, model, new_size, attribute_name, max_trials)
        else:
            raise ValueError('mode in method `scale_batch_size` can only be `power` or `binsearch')
        garbage_collection_cuda()

        # Convert times to work on same data amount
        max_batch_size = max(bs for bs, suc in \
                             zip(batch_scaler.results['batch_size'], batch_scaler.results['fits_in_memory']) if suc)
        batch_scaler.results['time'] = [t * max_batch_size/bs for t,bs in \
                                        zip(batch_scaler.results['time'], batch_scaler.results['batch_size'])]
        
        # Restore initial state of model
        self.trainer.restore(str(save_path), on_gpu=self.trainer.on_gpu)
        os.remove(save_path)

        # Finish by resetting variables so trainer is ready to fit model
        self.__scale_batch_restore_params()
        if self.trainer.progress_bar_callback:
            self.trainer.progress_bar_callback.enable()
        
        # Log that method was called and return object
        self._scale_batch_size_called = True
        return batch_scaler

    def __scale_batch_dump_params(self):
        # Prevent going into infinite loop
        self.__dumped_params = {
            'max_steps': self.trainer.max_steps,
            'weights_summary': self.trainer.weights_summary,
            'logger': self.trainer.logger,
            'callbacks': self.trainer.callbacks,
            'checkpoint_callback': self.trainer.checkpoint_callback,
            'early_stop_callback': self.trainer.early_stop_callback,
            'enable_early_stop': self.trainer.enable_early_stop,
            'train_percent_check': self.trainer.train_percent_check,
        }

    def __scale_batch_reset_params(self, model, steps_per_trial):
        self.trainer.max_steps = steps_per_trial  # take few steps
        self.trainer.weights_summary = None  # not needed before full run
        self.trainer.logger = DummyLogger()
        self.trainer.callbacks = []  # not needed before full run
        self.trainer.checkpoint_callback = False  # required for saving
        self.trainer.early_stop_callback = None
        self.trainer.enable_early_stop = False
        self.trainer.train_percent_check = 1.0
        self.trainer.optimizers, self.trainer.schedulers = [], []  # required for saving
        self.trainer.model = model  # required for saving

    def __scale_batch_restore_params(self):
        self.trainer.max_steps = self.__dumped_params['max_steps']
        self.trainer.weights_summary = self.__dumped_params['weights_summary']
        self.trainer.logger = self.__dumped_params['logger']
        self.trainer.callbacks = self.__dumped_params['callbacks']
        self.trainer.checkpoint_callback = self.__dumped_params['checkpoint_callback']
        self.trainer.early_stop_callback = self.__dumped_params['early_stop_callback']
        self.trainer.enable_early_stop = self.__dumped_params['enable_early_stop']
        self.trainer.train_percent_check = self.__dumped_params['train_percent_check']
        del self.__dumped_params

class BatchScaler(object):
    def __init__(self):
        self.results = {'batch_size': [], 'time': [], 'fits_in_memory': []}
    
    def plot(self, suggest=True, show=False):
        """ Plot results from batch_size_scaler run
        Args:
            suggest: if True, will mark suggested lr to use with a red point

            show: if True, will show figure
        """
        import matplotlib.pyplot as plt

        bs = np.array(self.results["batch_size"])
        times = np.array(self.results["time"])
        succes = np.array(self.results["fits_in_memory"])
        max_bs = np.max(bs[succes])

        # Reorder
        idx_sort = np.argsort(bs)
        bs, times, succes = bs[idx_sort], times[idx_sort], succes[idx_sort]
        
        # Plot time as function of batch size, mark largest batch size
        fig, ax = plt.subplots()
        ax.plot(bs, times, '-o')
        ax.set_yscale("log")
        ax.axvline(x=max_bs, ymin=0, ymax=max(times), color='green', label='maximum batch size')
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Time")
        
        # Plot suggestion
        if suggest:
            suggestion = self.suggestion()
            ax.plot(suggestion, self.results["time"][self._optimal_idx],
                    markersize=10, marker='o', color='red', label='suggestion')
        ax.legend()
        
        if show:
            plt.show()

        return fig
        
    def suggestion(self, condition='size'):
        """ This will propose a suggestion for choice of batch size either base
            on choosing the largest batch that fits in memory (default) or the
            batch size that approximately give the fastest training time.
        Args:
            condition: either `size` or `speed`
        Returns:
            lr: suggested batch size to use
        """
        assert condition in ['size', 'speed'], \
            'condition needs to be either `size` or `speed`'
        if condition == 'size':
            bs = np.array(self.results["batch_size"]).astype('int')
            suc = np.array(self.results["fits_in_memory"]).astype('int')
            self._optimal_idx = np.argmax(bs * suc)
            return self.results["batch_size"][self._optimal_idx]
        else:
            time = np.array(self.results["time"]).astype('float')
            suc = np.array(self.results["fits_in_memory"]).astype('float')
            self._optimal_idx = np.argmin(time * (1-suc) * 10**6)
            return self.results["batch_size"][self._optimal_idx]

def _adjust_batch_size(trainer,
                       batch_arg_name: str = 'batch_size',
                       factor: float = 1.0,
                       value: Optional[int] = None,
                       desc: str = None):
    """ Function for adjusting the batch size. It is expected that the user
        has provided a model that has a hparam field called `batch_size` i.e.
        `model.hparams.batch_size` should exist.

    Args:
        trainer: instance of pytorch_lightning.Trainer

        batch_arg_name: field where batch_size is stored in `model.hparams`

        factor: value which the old batch size is multiplied by to get the
            new batch size

        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case

        desc: either `succeeded` or `failed`. Used purely for logging
    """
    model = trainer.get_model()
    batch_size = lightning_getattr(model, batch_arg_name)
    if value:
        lightning_setattr(model, batch_arg_name, value)
        new_size = value
        if desc:
            log.info(f'Batch size {batch_size} {desc}, trying batch size {new_size}')
    else:
        new_size = int(batch_size * factor)
        if desc:
            log.info(f'Batch size {batch_size} {desc}, trying batch size {new_size}')
        lightning_setattr(model, batch_arg_name, new_size)
    return new_size


def _run_power_scaling(trainer, model, new_size, batch_arg_name, max_trials):
    """ Batch scaling mode where the size is doubled at each iteration until an
        OOM error is encountered. """
    batch_scaler = BatchScaler()
    for _ in range(max_trials):
        garbage_collection_cuda()
        trainer.global_step = 0  # reset after each try
        start = time.monotonic()
        batch_scaler.results['batch_size'].append(new_size)
        try:
            # Try fit
            trainer.fit(model)
            batch_scaler.results['fits_in_memory'].append(True)
            # Double in size
            new_size = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc='succeeded')
        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                batch_scaler.results['fits_in_memory'].append(False)    
                new_size = _adjust_batch_size(trainer, batch_arg_name, factor=0.5, desc='failed')
                end = time.monotonic()
                batch_scaler.results['time'].append(end-start)
                break
            else:
                raise  # some other error not memory related
        end = time.monotonic()
        batch_scaler.results['time'].append(end-start)
    return batch_scaler


def _run_binsearch_scaling(trainer, model, new_size, batch_arg_name, max_trials):
    """ Batch scaling mode where the size is initially is doubled at each iteration
        until an OOM error is encountered. Hereafter, the batch size is further
        refined using a binary search """
    batch_scaler = BatchScaler()
    high = None
    count = 0
    while True:
        garbage_collection_cuda()
        trainer.global_step = 0  # reset after each try
        start = time.monotonic()
        batch_scaler.results['batch_size'].append(new_size)
        try:
            # Try fit
            trainer.fit(model)
            batch_scaler.results['fits_in_memory'].append(True)
            count += 1
            if count > max_trials:
                end = time.monotonic()
                batch_scaler.results['time'].append(end-start)
                break
            # Double in size
            low = new_size
            if high:
                if high - low <= 1:
                    end = time.monotonic()
                    batch_scaler.results['time'].append(end-start)
                    break
                midval = (high + low) // 2
                new_size = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc='succeeded')
            else:
                new_size = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc='succeeded')
        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                batch_scaler.results['fits_in_memory'].append(False)
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                high = new_size
                midval = (high + low) // 2
                new_size = _adjust_batch_size(trainer, value=midval, desc='failed')
                if high - low <= 1:
                    end = time.monotonic()
                    batch_scaler.results['time'].append(end-start)
                    break
            else:
                raise  # some other error not memory related
        end = time.monotonic()
        batch_scaler.results['time'].append(end-start)
    
    return batch_scaler