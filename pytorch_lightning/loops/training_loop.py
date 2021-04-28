from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.epoch_loop import BatchLoop
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden


class TrainingLoop(Loop):
    """ Runs over all batches in a dataloader (one epoch). """

    def connect(self, trainer: 'pl.Trainer', *args, **kwargs):
        self.trainer = trainer
        self.batch_loop = BatchLoop()

    def on_run_start(self):
        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)

        self._train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        self._dataloader_idx = 0

    def advance(self):
        # TODO: profiling is gone
        batch_idx, (batch, is_last) = next(self._train_dataloader)

        self.trainer.batch_idx = batch_idx
        self.trainer.is_last_batch = is_last

        # ------------------------------------
        # TRAINING_STEP + TRAINING_STEP_END
        # ------------------------------------
        with self.trainer.profiler.profile("run_training_batch"):
            # batch_output = self.run_training_batch(batch, batch_idx, self._dataloader_idx)
            batch_output = self.batch_loop.run(batch, batch_idx, self._dataloader_idx)

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

    # this is the old on train_epoch_end?
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