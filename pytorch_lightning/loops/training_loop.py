from copy import deepcopy
from typing import List, Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.batch_loop import BatchLoop
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden


class TrainingLoop(Loop):
    """ Runs over all batches in a dataloader (one epoch). """

    def __init__(self):
        super().__init__()
        # cache of all outputs in a single training run / epoch
        # self.epoch_output = [[]]

    def connect(self, trainer: 'pl.Trainer', *args, **kwargs):
        self.trainer = trainer
        # self.epoch_output = [[] for _ in range(len(trainer.optimizers))]
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
        epoch_output = [[]]  # TODO: track and return output, let loop base concatenate all outputs into a list etc.
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

# ------------------------------------------------------------------------------------------------------------
# HELPER --- TO BE CLEANED UP
# ------------------------------------------------------------------------------------------------------------

    # TODO move to on_advance_end()
    def on_train_batch_end(self, epoch_output, batch_end_outputs, batch, batch_idx, dataloader_idx):
        batch_end_outputs = [opt_idx_out for opt_idx_out in batch_end_outputs if len(opt_idx_out)]

        processed_batch_end_outputs = self._prepare_outputs(batch_end_outputs, batch_mode=True)

        # hook
        self.trainer.call_hook('on_train_batch_end', processed_batch_end_outputs, batch, batch_idx, dataloader_idx)
        self.trainer.call_hook('on_batch_end')

        # figure out what to track for epoch end
        self.track_epoch_end_reduce_metrics(epoch_output, batch_end_outputs)

        # reset batch logger internals
        self.trainer.logger_connector.on_train_batch_end()

    def track_epoch_end_reduce_metrics(self, epoch_output, batch_end_outputs):

        # track the outputs to reduce at the end of the epoch
        for opt_idx, opt_outputs in enumerate(batch_end_outputs):
            sample_output = opt_outputs[-1]

            # decide if we need to reduce at the end of the epoch automatically
            auto_reduce_tng_result = isinstance(sample_output, Result) and sample_output.should_reduce_on_epoch_end
            hook_overridden = (
                is_overridden("training_epoch_end", model=self.trainer.lightning_module)
                or is_overridden("on_train_epoch_end", model=self.trainer.lightning_module)
            )

            # only track when a) it needs to be autoreduced OR b) the user wants to manually reduce on epoch end
            if not (hook_overridden or auto_reduce_tng_result):
                continue

            # with 1 step (no tbptt) don't use a sequence at epoch end
            if isinstance(opt_outputs, list) and len(opt_outputs) == 1 and not isinstance(opt_outputs[0], Result):
                opt_outputs = opt_outputs[0]

            epoch_output[opt_idx].append(opt_outputs)

    @staticmethod
    def _prepare_outputs(
            outputs: List[List[List[Result]]],
            batch_mode: bool,
    ) -> Union[List[List[List[Dict]]], List[List[Dict]], List[Dict], Dict]:
        """
        Extract required information from batch or epoch end results.

        Args:
            outputs: A 3-dimensional list of ``Result`` objects with dimensions:
                [optimizer outs][batch outs][tbptt steps].

            batch_mode: If True, ignore the batch output dimension.

        Returns:
            The cleaned outputs with ``Result`` objects converted to dictionaries. All list dimensions of size one will
            be collapsed.
        """
        processed_outputs = []
        for opt_outputs in outputs:
            # handle an edge case where an optimizer output is the empty list
            if len(opt_outputs) == 0:
                continue

            processed_batch_outputs = []

            if batch_mode:
                opt_outputs = [opt_outputs]

            for batch_outputs in opt_outputs:
                processed_tbptt_outputs = []

                for tbptt_output in batch_outputs:
                    out = tbptt_output.extra
                    out['loss'] = tbptt_output.minimize
                    processed_tbptt_outputs.append(out)

                # if there was only one tbptt step then we can collapse that dimension
                if len(processed_tbptt_outputs) == 1:
                    processed_tbptt_outputs = processed_tbptt_outputs[0]
                processed_batch_outputs.append(processed_tbptt_outputs)

            # batch_outputs should be just one dict (or a list of dicts if using tbptt) per optimizer
            if batch_mode:
                processed_batch_outputs = processed_batch_outputs[0]
            processed_outputs.append(processed_batch_outputs)

        # if there is only one optimiser then we collapse that dimension
        if len(processed_outputs) == 1:
            processed_outputs = processed_outputs[0]
        return processed_outputs

    def update_train_loop_lr_schedulers(self, monitor_metrics=None):
        num_accumulated_batches_reached = self._accumulated_batches_reached()
        num_training_batches_reached = self._num_training_batches_reached()

        if num_accumulated_batches_reached or num_training_batches_reached:
            # update lr
            self.trainer.optimizer_connector.update_learning_rates(interval="step", monitor_metrics=monitor_metrics)

    def increment_accumulated_grad_global_step(self):
        num_accumulated_batches_reached = self._accumulated_batches_reached()
        num_training_batches_reached = self._num_training_batches_reached()

        # progress global step according to grads progress
        if num_accumulated_batches_reached or num_training_batches_reached:
            self.trainer.global_step = self.trainer.accelerator.update_global_step(
                self.trainer.total_batch_idx, self.trainer.global_step
            )

    def should_check_val_fx(self, batch_idx, is_last_batch, on_epoch=False):
        # decide if we should run validation
        is_val_check_batch = (batch_idx + 1) % self.trainer.val_check_batch == 0
        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        can_check_val = self.trainer.enable_validation and is_val_check_epoch
        is_last_batch_for_infinite_dataset = is_last_batch and self.trainer.val_check_batch == float("inf")
        epoch_end_val_check = (batch_idx + 1) % self.trainer.num_training_batches == 0

        should_check_val = ((is_val_check_batch and epoch_end_val_check) or self.trainer.should_stop
                            or is_last_batch_for_infinite_dataset
                            ) if on_epoch else (is_val_check_batch and not epoch_end_val_check)

        return should_check_val and can_check_val

    def save_loggers_on_train_batch_end(self):
        # when loggers should save to disk
        should_flush_logs = self.trainer.logger_connector.should_flush_logs
        if should_flush_logs and self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.save()