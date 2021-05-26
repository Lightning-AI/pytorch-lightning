import logging
from contextlib import suppress
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.training_loop import TrainingLoop
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.warnings import WarningCache

log = logging.getLogger(__name__)


class EpochLoop(Loop):

    def __init__(self, min_epochs, max_epochs, min_steps, max_steps):
        super().__init__()
        self._teardown_already_run = False

        # If neither max_epochs or max_steps is set, then use existing default of max_epochs = 1000
        self.max_epochs = 1000 if (max_epochs is None and max_steps is None) else max_epochs
        # If neither min_epochs or min_steps is set, then use existing default of min_epochs = 1
        self.min_epochs = 1 if (min_epochs is None and min_steps is None) else min_epochs

        self.training_loop = TrainingLoop(min_steps, max_steps)

    @property
    def current_epoch(self) -> int:
        return self.iteration_count

    @current_epoch.setter
    def current_epoch(self, value: int):
        self.iteration_count = value

    @property
    def global_step(self):
        return self.training_loop.global_step

    @global_step.setter
    def global_step(self, value):
        self.training_loop.global_step = value

    @property
    def total_batch_idx(self):
        return self.training_loop.total_batch_idx

    @property
    def batch_idx(self):
        return self.training_loop.iteration_count

    @property
    def split_idx(self):
        return self.training_loop.split_idx

    @property
    def min_steps(self):
        return self.training_loop.min_steps

    @property
    def max_steps(self):
        return self.training_loop.max_steps

    @max_steps.setter
    def max_steps(self, value):
        # TODO: This setter is required by debugging connector (fast dev run)
        self.training_loop.max_steps = value

    @property
    def running_loss(self):
        return self.training_loop.batch_loop.running_loss

    @property
    def skip_backward(self) -> bool:
        """ Determines whether the loop will skip backward during automatic optimization. """
        return self.training_loop.batch_loop.skip_backward

    @skip_backward.setter
    def skip_backward(self, value: bool):
        """ Determines whether the loop will skip backward during automatic optimization. """
        self.training_loop.batch_loop.skip_backward = value

    def connect(self, trainer: 'pl.Trainer', *args, **kwargs):
        self.trainer = trainer
        self.training_loop.connect(trainer)

    @property
    def done(self) -> bool:
        # TODO: Move track steps inside training loop and move part of these condition inside training loop
        stop_steps = self.max_steps and self.max_steps <= self.global_step

        should_stop = False
        if self.trainer.should_stop:
            # early stopping
            met_min_epochs = (self.current_epoch >= self.min_epochs - 1) if self.min_epochs else True
            met_min_steps = self.global_step >= self.min_steps if self.min_steps else True
            if met_min_epochs and met_min_steps:
                # TODO: THIS is now in on_run_end, always run?
                # self.training_loop.on_train_end()
                should_stop = True
            else:
                log.info(
                    'Trainer was signaled to stop but required minimum epochs'
                    f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                    ' not been met. Training will continue...'
                )
                self.trainer.should_stop = False

        stop_epochs = self.current_epoch >= self.max_epochs if self.max_epochs is not None else False
        return stop_steps or should_stop or stop_epochs

    def on_run_start(self):
        # hook
        self.trainer.call_hook("on_train_start")

    def on_advance_start(self):  # equal to old on_train_epoch_start
        model = self.trainer.lightning_module

        # reset train dataloader
        if self.current_epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.training_loop.batch_loop.accumulated_loss = TensorRunningAccum(
            window_length=self.trainer.accumulate_grad_batches
        )

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

    def advance(self):

        with self.trainer.profiler.profile("run_training_epoch"):
            # run train epoch
            epoch_output = self.training_loop.run()
            # log epoch metrics

            if epoch_output is None:
                return

            self.trainer.logger_connector.log_train_epoch_end_metrics(epoch_output)

    def on_advance_end(self):
        # # handle epoch_output on epoch end
        # self.on_train_epoch_end(outputs)  # Handled in on_run_end of training_loop now

        if self.training_loop.batches_seen == 0:
            return

        should_check_val = self.training_loop.should_check_val_fx(
            self.batch_idx, self.training_loop.is_last_batch, on_epoch=True
        )
        should_skip_eval = self.trainer.evaluation_loop.should_skip_evaluation(self.trainer.num_val_batches)
        should_train_only = self.trainer.disable_validation or should_skip_eval

        # update epoch level lr_schedulers if no val loop outside train loop is triggered
        if not should_check_val or should_train_only:
            self.training_loop.update_lr_schedulers("epoch")

        if should_train_only:
            self.check_checkpoint_callback(True)

        if should_check_val:
            self.trainer.validating = True
            self.trainer._run_evaluation(on_epoch=True)
            self.trainer.training = True

        # TODO: move inside training_loop.on_run_end? equivalent? order?
        #   Needs to check batch_output signal -1, see #7677
        self.training_loop.increment_accumulated_grad_global_step()

    # why is this not the same as the old on_train_epoch_end?
    def on_run_end(self):
        if self._teardown_already_run:
            return
        self._teardown_already_run = True

        # NOTE: the iteration_count/current_epoch is already incremented
        # Lightning today does not increment the current epoch at the last epoch run in Trainer.fit
        # To simulate that current behavior, we decrement here.
        self.current_epoch -= 1

        # trigger checkpoint check. need to temporarily decrease the global step to avoid saving duplicates
        # when a checkpoint was saved at the last step
        self.training_loop.global_step -= 1
        # TODO: see discussion/rework https://github.com/PyTorchLightning/pytorch-lightning/issues/7406
        self.check_checkpoint_callback(should_update=True, is_last=True)
        self.training_loop.global_step += 1

        # hook
        self.trainer.call_hook("on_train_end")

        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu
        # kill loggers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize("success")

        # summarize profile results
        self.trainer.profiler.describe()

        # give accelerators a chance to finish
        self.trainer.accelerator.on_train_end()

        # reset bookkeeping
        self.trainer._running_stage = None

    def should_accumulate(self):
        return self.training_loop.batch_loop.should_accumulate()

    def get_active_optimizers(self, batch_idx: Optional[int] = None) -> List[Tuple[int, Optimizer]]:
        return self.training_loop.batch_loop.get_active_optimizers(batch_idx)

    def check_checkpoint_callback(self, should_update, is_last=False):
        # TODO bake this logic into the ModelCheckpoint callback
        if should_update and self.trainer.checkpoint_connector.has_trained:
            callbacks = self.trainer.checkpoint_callbacks

            if is_last and any(cb.save_last and cb.verbose for cb in callbacks):
                rank_zero_info("Saving latest checkpoint...")

            model = self.trainer.lightning_module

            for cb in callbacks:
                cb.on_validation_end(self.trainer, model)
