from contextlib import suppress
from copy import deepcopy
from logging import log
from typing import Any, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.supporters import TensorRunningAccum
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


class TrainingLoop(Loop):

    def connect(self, trainer: 'pl.Trainer'):
        self.trainer = trainer
        self.batch_loop = BatchLoop

    def on_run_start(self):
        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)

        self._train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        self._dataloader_idx = 0

    def advance(self):
        batch_idx, (batch, is_last) = next(self._train_dataloader)

        self.trainer.batch_idx = batch_idx
        self.trainer.is_last_batch = is_last

        # ------------------------------------
        # TRAINING_STEP + TRAINING_STEP_END
        # ------------------------------------
        with self.trainer.profiler.profile("run_training_batch"):
            batch_output = self.run_training_batch(batch, batch_idx, self._dataloader_idx)

        # when returning -1 from train_step, we end epoch early
        if batch_output.signal == -1:
            self._skip_remaining_steps = True
            return

        # hook
        # TODO: add outputs to batches
        self.on_train_batch_end(
            epoch_output,
            batch_output.training_step_output_for_epoch_end,
            batch,
            batch_idx,
            self._dataloader_idx,
        )

    def on_advance_end(self, output):
        # -----------------------------------------
        # SAVE METRICS TO LOGGERS
        # -----------------------------------------
        self.trainer.logger_connector.log_train_step_metrics(output)

        # -----------------------------------------
        # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
        # -----------------------------------------
        should_check_val = self.should_check_val_fx(self.trainer.batch_idx, self.trainer.is_last_batch)
        if should_check_val:
            self.trainer.validating = True
            self.trainer.run_evaluation()
            self.trainer.training = True

        # -----------------------------------------
        # SAVE LOGGERS (ie: Tensorboard, etc...)
        # -----------------------------------------
        self.save_loggers_on_train_batch_end()

        # update LR schedulers
        monitor_metrics = deepcopy(self.trainer.logger_connector.callback_metrics)
        self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)
        self.trainer.checkpoint_connector.has_trained = True

        # progress global step according to grads progress
        self.increment_accumulated_grad_global_step()

    @property
    def done(self):
        # max steps reached, end training
        if (
            self.trainer.max_steps is not None and self.trainer.max_steps <= self.trainer.global_step + 1
            and self._accumulated_batches_reached()
        ):
            return True

        # end epoch early
        # stop when the flag is changed or we've gone past the amount
        # requested in the batches
        if self.trainer.should_stop:
            return True

        self.trainer.total_batch_idx += 1

        # stop epoch if we limited the number of training batches
        if self._num_training_batches_reached(self.trainer.is_last_batch):
            return True

    def on_run_end(self, outputs):
        # inform logger the batch loop has finished
        self.trainer.logger_connector.on_train_epoch_end()

        # prepare epoch output
        processed_outputs = self._prepare_outputs(outputs, batch_mode=False)

        # get the model and call model.training_epoch_end
        model = self.trainer.lightning_module

        if is_overridden('training_epoch_end', model=model):
            # run training_epoch_end
            # refresh the result for custom logging at the epoch level
            model._current_fx_name = 'training_epoch_end'

            # lightningmodule hook
            training_epoch_end_output = model.training_epoch_end(processed_outputs)

            if training_epoch_end_output is not None:
                raise MisconfigurationException(
                    'training_epoch_end expects a return of None. '
                    'HINT: remove the return statement in training_epoch_end'
                )

            # capture logging
            self.trainer.logger_connector.cache_logged_metrics()

        # call train epoch end hooks
        self.trainer.call_hook('on_train_epoch_end', processed_outputs)
        self.trainer.call_hook('on_epoch_end')

        # increment the global step once
        # progress global step according to grads progress
        self.increment_accumulated_grad_global_step()


class BatchLoop(Loop):

    def on_run_start(self, batch, batch_idx, dataloader_idx):
        self._grad_norm_dic = {}
        self.trainer.hiddens = None
        self._optimizers = self.prepare_optimizers()
        # lightning module hook
        self._splits = self.tbptt_split_batch(batch)

    def on_advance_start(self):
        return super().on_advance_start()

    def advance(self, *args: Any, **kwargs: Any):
        return super().advance(*args, **kwargs)

    def run(self, batch, batch_idx, dataloader_idx):
        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic={})

        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic={})

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic={})

        return super().run(batch, batch_idx, dataloader_idx)


def run_training_batch(self, batch, batch_idx, dataloader_idx):

    for split_idx, split_batch in enumerate(splits):

        # create an iterable for optimizers and loop over them
        for opt_idx, optimizer in optimizers:

            # toggle model params + set info to logger_connector
            self.run_train_split_start(split_idx, split_batch, opt_idx, optimizer)

            if self.should_accumulate():
                # For gradient accumulation

                # -------------------
                # calculate loss (train step + train step end)
                # -------------------

                # automatic_optimization=True: perform dpp sync only when performing optimizer_step
                # automatic_optimization=False: don't block synchronization here
                with self.block_ddp_sync_behaviour():
                    self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens)

                batch_outputs = self._process_closure_result(
                    batch_outputs=batch_outputs,
                    opt_idx=opt_idx,
                )

            # ------------------------------
            # BACKWARD PASS
            # ------------------------------
            # gradient update with accumulated gradients

            else:
                if self.automatic_optimization:

                    def train_step_and_backward_closure():
                        result = self.training_step_and_backward(
                            split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
                        )
                        return None if result is None else result.loss

                    # optimizer step
                    self.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)

                else:
                    self._curr_step_result = self.training_step(split_batch, batch_idx, opt_idx, self.trainer.hiddens)

                if self._curr_step_result is None:
                    # user decided to skip optimization
                    # make sure to zero grad.
                    continue

                batch_outputs = self._process_closure_result(
                    batch_outputs=batch_outputs,
                    opt_idx=opt_idx,
                )

                # todo: Properly aggregate grad_norm accros opt_idx and split_idx
                grad_norm_dic = self._cur_grad_norm_dict
                self._cur_grad_norm_dict = None

                # update running loss + reset accumulated loss
                self.update_running_loss()

    result = AttributeDict(
        signal=0,
        grad_norm_dic=grad_norm_dic,
        training_step_output_for_epoch_end=batch_outputs,
    )
    return result
