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

from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, List
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer.training_loop_temp import TrainLoop
from pytorch_lightning.trainer.data_connector import DataConnector
from pytorch_lightning.trainer.logger_connector import LoggerConnector


class TrainerTrainLoopMixin(ABC):
    on_gpu: bool
    use_horovod: bool
    check_val_every_n_epoch: ...
    num_training_batches: int
    val_check_batch: ...
    fast_dev_run: ...
    lr_schedulers: ...
    callback_metrics: ...
    logger: Union[LightningLoggerBase, bool]
    global_step: int
    log_save_interval: float
    row_log_interval: float
    truncated_bptt_steps: ...
    optimizers: ...
    accumulate_grad_batches: int
    model: LightningModule
    running_loss: ...
    profiler: ...
    batch_idx: int
    max_steps: int
    terminate_on_nan: bool
    _state: TrainerState
    accelerator_backend: ...
    train_loop: TrainLoop
    data_connector: DataConnector
    logger_connector: LoggerConnector

    # Callback system
    callbacks: List[Callback]
    on_batch_start: Callable
    on_train_batch_start: Callable
    on_train_batch_end: Callable
    on_epoch_end: Callable
    on_validation_end: Callable
    on_train_epoch_end: Callable

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def detect_nan_tensors(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def process_output(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def call_hook(self, hook_name, *args, **kwargs):
        """Warning: this is just empty shell for code implemented in other class."""

    def process_train_step_outputs(self, all_train_step_outputs, early_stopping_accumulator, checkpoint_accumulator):
        """
        Figure out what needs to be tracked/logged at the end of the epoch
        """

        # the training step outputs a list per optimizer. The list contains the outputs at each time step
        # when no TBPTT is used, then the list has 1 item per batch
        # when TBPTT IS used, then the list has n items (1 per time step)
        epoch_end_outputs = []
        for optimizer_idx_outputs in all_train_step_outputs:
            # extract one representative sample from each time step (1 if no tbptt) and 0th optimizer
            sample_output = optimizer_idx_outputs[-1]

            # pull out callback info if available (ie: Results object)
            if isinstance(sample_output, dict) and 'early_stop_on' in sample_output:
                early_stopping_accumulator.accumulate(sample_output['early_stop_on'])

            if isinstance(sample_output, dict) and 'checkpoint_on' in sample_output:
                checkpoint_accumulator.accumulate(sample_output['checkpoint_on'])

            # decide if we need to reduce at the end of the epoch automatically
            auto_reduce_tng_result = isinstance(sample_output, Result) and sample_output.should_reduce_on_epoch_end

            # only track when a) it needs to be autoreduced OR b) the user wants to manually reduce on epoch end
            if is_overridden('training_epoch_end', model=self.get_model()) or auto_reduce_tng_result:
                epoch_end_outputs.append(optimizer_idx_outputs)

        return epoch_end_outputs

    def run_training_epoch_end(self, epoch_output, checkpoint_accumulator, early_stopping_accumulator, num_optimizers):
        # epoch output is a list. Each item in that list has all the outputs per optimizer
        # epoch_output[optimizer_idx][training_step_idx][tbptt_index]
        # remember that not using truncated backprop is equivalent with truncated back prop of len(1)

        model = self.get_model()

        epoch_log_metrics = {}
        epoch_callback_metrics = {}
        epoch_progress_bar_metrics = {}

        # -----------------------
        # Calculate epoch callback values if given
        # -----------------------
        if checkpoint_accumulator.num_values > 0:
            epoch_callback_metrics['checkpoint_on'] = checkpoint_accumulator.mean()

        if early_stopping_accumulator.num_values > 0:
            epoch_callback_metrics['early_stop_on'] = early_stopping_accumulator.mean()

        # ------------------------
        # determine if using a result obj
        # ------------------------
        # [optimizer_idx][training_step_idx][tbptt_index]
        opt_idx_outputs = epoch_output[0]

        try:
            sample_obj = opt_idx_outputs[0][0] if isinstance(opt_idx_outputs[0], list) else opt_idx_outputs[0]
            is_result_obj = len(epoch_output) > 0 and isinstance(sample_obj, Result)
        except IndexError as e:
            is_result_obj = False

        # --------------------------
        # EPOCH END STEP IF DEFINED
        # --------------------------
        if is_overridden('training_epoch_end', model=model):
            self.global_step += 1

            if is_result_obj:
                # with result object gather across time and training steps so each opt idx has a single result obj
                epoch_output = self.__gather_result_across_time_and_optimizers(epoch_output)

            if num_optimizers == 1:
                epoch_output = epoch_output[0]

            # run training_epoch_end
            # a list with a result per optimizer index
            epoch_output = model.training_epoch_end(epoch_output)

            if isinstance(epoch_output, Result):
                epoch_log_metrics = epoch_output.epoch_log_metrics
                epoch_progress_bar_metrics = epoch_output.epoch_pbar_metrics
            else:
                _processed_outputs = self.process_output(epoch_output)
                epoch_progress_bar_metrics = _processed_outputs[1]
                epoch_log_metrics = _processed_outputs[2]
                epoch_callback_metrics = _processed_outputs[3]

        # --------------------------
        # Structured Result (auto epoch end)
        # --------------------------
        elif is_result_obj:
            epoch_log_metrics, epoch_progress_bar_metrics = self.__auto_reduce_results_on_epoch_end(epoch_output)

        # --------------------------
        # track results
        # --------------------------
        # add the metrics to the loggers
        if epoch_log_metrics and len(epoch_log_metrics) > 0:
            self.logger_connector.log_metrics(epoch_log_metrics, {})

        # add metrics to callbacks
        self.logger_connector.callback_metrics.update(epoch_callback_metrics)

        # add metrics to progress_bar
        if len(epoch_progress_bar_metrics) > 0:
            self.logger_connector.add_progress_bar_metrics(epoch_progress_bar_metrics)

    def __auto_reduce_results_on_epoch_end(self, epoch_output):
        epoch_log_metrics = {}
        epoch_progress_bar_metrics = {}
        for opt_outputs in epoch_output:
            # reduce across time first
            time_reduced_outputs = []
            for train_step_idx in range(len(opt_outputs)):
                tbptt_outs = opt_outputs[train_step_idx]
                tbptt_outs = tbptt_outs[0].__class__.reduce_across_time(tbptt_outs)
                time_reduced_outputs.append(tbptt_outs)

            # reduce across training steps
            opt_outputs = time_reduced_outputs[0].__class__.reduce_on_epoch_end(time_reduced_outputs)
            opt_outputs.minimize = opt_outputs.minimize.mean()
            epoch_log_metrics.update(opt_outputs.epoch_log_metrics)
            epoch_progress_bar_metrics.update(opt_outputs.epoch_pbar_metrics)

        return epoch_log_metrics, epoch_progress_bar_metrics

    def __gather_result_across_time_and_optimizers(self, epoch_output):
        """
        Gather results into a single padded tensor per metric where each tensor is gathered across
        time and across time steps.

        Returns:
            a list where each element is a Result with the tensors gathered
        """
        gathered_epoch_outputs = []
        for opt_outputs in epoch_output:
            # gather across time first
            time_gathered_outputs = []
            for train_step_idx in range(len(opt_outputs)):
                tbptt_outs = opt_outputs[train_step_idx]
                tbptt_outs = tbptt_outs[0].__class__.gather(tbptt_outs)
                time_gathered_outputs.append(tbptt_outs)

            # gather across training steps
            # each metric has dimensions (training_steps, seq_len) (seq_len=1 when no tbptt is used)
            gathered_opt_output = time_gathered_outputs[0].__class__.padded_gather(time_gathered_outputs)
            gathered_epoch_outputs.append(gathered_opt_output)

        return gathered_epoch_outputs

    def save_train_loop_metrics_to_loggers(self, batch_idx, batch_output):
        # when metrics should be logged
        should_log_metrics = (batch_idx + 1) % self.row_log_interval == 0 or self.should_stop
        if should_log_metrics or self.fast_dev_run:
            # logs user requested information to logger
            metrics = batch_output.batch_log_metrics
            grad_norm_dic = batch_output.grad_norm_dic
            if len(metrics) > 0 or len(grad_norm_dic) > 0:
                self.logger_connector.log_metrics(metrics, grad_norm_dic)

    def save_loggers_in_training_loop(self, batch_idx):
        # when loggers should save to disk
        should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or self.should_stop
        if should_save_log or self.fast_dev_run:
            if self.is_global_zero and self.logger is not None:
                self.logger.save()

    def run_training_batch(self, batch, batch_idx, dataloader_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        batch_callback_metrics = []

        # track metrics to log
        batch_log_metrics = []

        # bookkeeping
        using_results_obj = False
        self.hiddens = None

        # track all outputs across time and num of optimizers
        batch_outputs = [[] for _ in range(len(self.train_loop.get_optimizers_iterable()))]

        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic=grad_norm_dic)

        # hook
        response = self.call_hook('on_batch_start')
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        # hook
        response = self.call_hook('on_train_batch_start', batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        # lightning module hook
        splits = self.train_loop.tbptt_split_batch(batch)

        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            # loop over optimizers
            for opt_idx, optimizer in self.train_loop.get_optimizers_iterable():
                # make sure only the gradients of the current optimizer's parameters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                # -------------------
                # calculate loss (train step + train step end)
                # -------------------
                opt_closure_result = self.train_loop.training_step_and_backward(
                    split_batch,
                    batch_idx,
                    opt_idx,
                    optimizer,
                    self.hiddens
                )

                # log metrics
                self.train_loop.log_training_step_metrics(opt_closure_result, batch_callback_metrics, batch_log_metrics)

                # track hiddens
                self.hiddens = self.train_loop.process_hiddens(opt_closure_result)

                # check if loss or model weights are nan
                if self.terminate_on_nan:
                    self.detect_nan_tensors(opt_closure_result.loss)

                # track total loss for logging (avoid mem leaks)
                self.train_loop.accumulated_loss.append(opt_closure_result.loss)

                # track all the outputs across all steps
                batch_outputs[opt_idx].append(opt_closure_result.training_step_output_for_epoch_end)

                # ------------------------------
                # BACKWARD PASS
                # ------------------------------
                # gradient update with accumulated gradients
                accumulation_done = (self.batch_idx + 1) % self.accumulate_grad_batches == 0
                is_final_batch = (self.batch_idx + 1) == self.num_training_batches
                if accumulation_done or is_final_batch:
                    # hook
                    grad_norm_dic = self.train_loop.on_before_backward(batch_idx, optimizer)

                    # wrap forward + backward pass in closure for 2nd order optimizers
                    train_step_and_backward_closure = lambda: self.train_loop.training_step_and_backward(
                        split_batch, batch_idx, opt_idx, optimizer, self.hiddens,
                    ).loss

                    # optimizer step
                    self.train_loop.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)

                    # hook
                    self.train_loop.on_before_zero_grad(optimizer)

                    # clear gradients
                    self.train_loop.optimizer_zero_grad(batch_idx, optimizer, opt_idx)

                    # calculate running loss for display
                    self.running_loss.append(self.train_loop.accumulated_loss.mean() * self.accumulate_grad_batches)

                    # reset for next set of accumulated grads
                    self.train_loop.accumulated_loss.reset()

        # collapse all metrics into one dict
        batch_log_metrics = {k: v for d in batch_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        if not using_results_obj:
            self.logger_connector.callback_metrics.update({k: v for d in batch_callback_metrics for k, v in d.items()})

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            batch_log_metrics=batch_log_metrics,
            training_step_output_for_epoch_end=batch_outputs
        )
        return result
