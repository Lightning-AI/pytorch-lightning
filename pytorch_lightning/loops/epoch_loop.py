from contextlib import suppress
from copy import deepcopy
from logging import log
from typing import Any, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.warnings import WarningCache


class EpochLoop(Loop):

    def connect(
        self,
        num_epochs: int,
        max_steps: Optional[int],
        trainer: 'pl.Trainer',
        *loops_to_run: Loop,
    ):
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.trainer = trainer
        self.loops_to_run = []
        for loop in loops_to_run:
            if isinstance(loop, Loop) or hasattr(loop, 'run'):
                self.loops_to_run.append(loop)

    @property
    def done(self) -> bool:
        stop_steps = self.trainer.max_steps and self.trainer.max_steps <= self.trainer.global_step

        should_stop = False
        if self.trainer.should_stop:
            # early stopping
            met_min_epochs = (self.iteration_count >= self.trainer.min_epochs - 1) if self.trainer.min_epochs else True
            met_min_steps = self.trainer.global_step >= self.trainer.min_steps if self.trainer.min_steps else True
            if met_min_epochs and met_min_steps:
                self.train_loop.on_train_end()
                should_stop = True
            else:
                log.info(
                    'Trainer was signaled to stop but required minimum epochs'
                    f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                    ' not been met. Training will continue...'
                )
                self.trainer.should_stop = False

        stop_epochs = self.iteration_count >= self.num_epochs

        return stop_steps or should_stop or stop_epochs

    def on_run_start(self):
        # hook
        self.trainer.call_hook("on_train_start")

    def on_run_end(self):
        if self._teardown_already_run:
            return
        self._teardown_already_run = True

        # trigger checkpoint check. need to temporarily decrease the global step to avoid saving duplicates
        # when a checkpoint was saved at the last step
        self.trainer.global_step -= 1
        self.check_checkpoint_callback(should_update=True, is_last=True)
        self.trainer.global_step += 1

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

    def on_advance_start(self):  # equal to on train epoch start
        # implemented here since this code has to be run always no matter the actual epoch implementation
        epoch = self.iteration_count + 1

        # update training progress in trainer
        self.trainer.current_epoch = epoch

        model = self.trainer.lightning_module

        # reset train dataloader
        if epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

    # why is this not the same as the old on_train_epoch_end?
    def on_advance_end(self, outputs):
        # handle epoch_output on epoch end
        self.on_train_epoch_end(outputs)

        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(outputs)

        should_check_val = self.should_check_val_fx(self.trainer.batch_idx, self.trainer.is_last_batch, on_epoch=True)
        should_skip_eval = self.trainer.evaluation_loop.should_skip_evaluation(self.trainer.num_val_batches)
        should_train_only = self.trainer.disable_validation or should_skip_eval

        # update epoch level lr_schedulers if no val loop outside train loop is triggered
        if (val_loop_called and not should_check_val) or should_train_only:
            self.trainer.optimizer_connector.update_learning_rates(interval='epoch')

        if should_train_only:
            self.check_checkpoint_callback(True)
            self.check_early_stopping_callback(True)

        if should_check_val:
            self.trainer.validating = True
            self.trainer.run_evaluation(on_epoch=True)
            self.trainer.training = True

        # increment the global step once
        # progress global step according to grads progress
        self.increment_accumulated_grad_global_step()

    def advance(self):
        ret_vals = []
        with self.trainer.profiler.profile("run_training_epoch"):
            # run train epoch
            for loop in self.loops_to_run:
                ret_vals.append(loop.run())

        return ret_vals

