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

"""
The lightning training loop handles everything except the actual computations of your model.
 To decide what will happen in your training loop, define the `training_step` function.

Below are all the things lightning automates for you in the training loop.

Accumulated gradients
---------------------

Accumulated gradients runs K small batches of size N before doing a backwards pass.
 The effect is a large effective batch size of size KxN.

.. code-block:: python

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)

Force training for min or max epochs
------------------------------------

It can be useful to force training for a minimum number of epochs or limit to a max number

.. code-block:: python

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)

Force disable early stop
------------------------

To disable early stopping pass None to the early_stop_callback

.. code-block:: python

    # DEFAULT
    trainer = Trainer(early_stop_callback=None)

Gradient Clipping
-----------------

Gradient clipping may be enabled to avoid exploding gradients.
 Specifically, this will `clip the gradient norm computed over all model parameters
 `together <https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_.

.. code-block:: python

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients with norm above 0.5
    trainer = Trainer(gradient_clip_val=0.5)

Inspect gradient norms
----------------------

Looking at grad norms can help you figure out where training might be going wrong.

.. code-block:: python

    # DEFAULT (-1 doesn't track norms)
    trainer = Trainer(track_grad_norm=-1)

    # track the LP norm (P=2 here)
    trainer = Trainer(track_grad_norm=2)

Set how much of the training set to check
-----------------------------------------

If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag.

limit_train_batches will be overwritten by overfit_batches if `overfit_batches > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(limit_train_batches=1.0)

    # check 10% only
    trainer = Trainer(limit_train_batches=0.1)

    # check 10 batches only
    trainer = Trainer(limit_train_batches=10)

Packed sequences as inputs
--------------------------

When using PackedSequence, do 2 things:
1. return either a padded tensor in dataset or a list of variable length tensors
in the dataloader collate_fn (example above shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

.. code-block:: python

    # For use in dataloader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y

    # In module
    def training_step(self, batch, batch_idx):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)


Truncated Backpropagation Through Time
--------------------------------------

There are times when multiple backwards passes are needed for each batch.
 For example, it may save memory to use Truncated Backpropagation Through Time when training RNNs.

When this flag is enabled each batch is split into sequences of size truncated_bptt_steps
 and passed to training_step(...) separately. A default splitting function is provided,
 however, you can override it for more flexibility. See `tbptt_split_batch`.

.. code-block:: python

    # DEFAULT (single backwards pass per batch)
    trainer = Trainer(truncated_bptt_steps=None)

    # (split batch into sequences of size 2)
    trainer = Trainer(truncated_bptt_steps=2)


NaN detection and intervention
------------------------------
When the `terminate_on_nan` flag is enabled, after every forward pass during training, Lightning will
check that

1. the loss you return in `training_step` is finite (not NaN and not +/-inf)
2. the model parameters have finite values.

Lightning will terminate the training loop with an error message if NaN or infinite
values are detected. If this happens, you should investigate numerically unstable operations
in your model.

.. code-block:: python

    # DEFAULT (won't perform the NaN check)
    trainer = Trainer(terminate_on_nan=False)

    # (NaN check each batch and terminate on NaN or infinite values)
    trainer = Trainer(terminate_on_nan=True)

"""
from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, List

from torch.utils.data import DataLoader
from copy import deepcopy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer.training_loop_temp import TrainLoop
from pytorch_lightning.trainer.data_connector import DataConnector
from pytorch_lightning.utilities.debugging import InternalDebugger


class TrainerTrainLoopMixin(ABC):
    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
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
    train_dataloader: DataLoader
    max_steps: int
    total_batch_idx: int
    terminate_on_nan: bool
    _state: TrainerState
    accelerator_backend: ...
    train_loop: TrainLoop
    data_connector: DataConnector
    dev_debugger: InternalDebugger

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
    def is_function_implemented(self, *args, **kwargs):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def run_evaluation(self, *args, **kwargs):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def detect_nan_tensors(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_progress_bar_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def log_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def process_output(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def call_hook(self, hook_name, *args, **kwargs):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def has_arg(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def run_training_epoch(self):

        # get model
        model = self.get_model()

        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.accelerator_backend.process_dataloader(self.train_dataloader)

        # track epoch output
        epoch_output = [[] for _ in range(self.train_loop.num_optimizers)]

        # enable profiling for the dataloader
        train_dataloader = self.data_connector.get_profiled_train_dataloader(train_dataloader)
        dataloader_idx = 0
        for batch_idx, (batch, is_last_batch) in train_dataloader:
            # stop epoch if we limited the number of training batches
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx
            model.global_step = self.global_step

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # only track outputs when user implements training_epoch_end
            # otherwise we will build up unnecessary memory
            epoch_end_outputs = self.process_train_step_outputs(
                batch_output.training_step_output_for_epoch_end,
                self.train_loop.early_stopping_accumulator,
                self.train_loop.checkpoint_accumulator
            )

            # hook
            self.train_loop.on_train_batch_end(epoch_output, epoch_end_outputs, batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            self.should_stop = batch_output.signal == -1

            # -----------------------------------------
            # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
            # -----------------------------------------
            should_check_val = self.should_check_val(batch_idx, is_last_batch)
            if should_check_val:
                self.run_evaluation(test_mode=False)

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_in_training_loop(batch_idx)

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.save_train_loop_metrics_to_loggers(batch_idx, batch_output)

            # update LR schedulers
            monitor_metrics = deepcopy(self.callback_metrics)
            monitor_metrics.update(batch_output.batch_log_metrics)
            self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

            # max steps reached, end training
            if self.max_steps is not None and self.max_steps == self.global_step:
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.should_stop:
                break

        # let ddp devices catch up when using horovod
        self.sync_horovod()

        # process epoch outputs
        self.run_training_epoch_end(
            epoch_output,
            self.train_loop.checkpoint_accumulator,
            self.train_loop.early_stopping_accumulator,
            self.train_loop.num_optimizers
        )

        # checkpoint callback
        self.check_checkpoint_callback(self.train_loop.should_check_val)

        # epoch end hook
        self.run_on_epoch_end_hook(model)

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

    def check_checkpoint_callback(self, should_check_val):
        # when no val loop is present or fast-dev-run still need to call checkpoints
        # TODO bake this logic into the checkpoint callback
        should_activate = not is_overridden('validation_step', self.get_model()) and not should_check_val
        if should_activate:
            checkpoint_callbacks = [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]
            [c.on_validation_end(self, self.get_model()) for c in checkpoint_callbacks]

    def update_train_loop_lr_schedulers(self, monitor_metrics=None):
        if ((self.batch_idx + 1) % self.accumulate_grad_batches == 0
                or (self.batch_idx + 1) == self.num_training_batches):
            # update lr
            self.update_learning_rates(interval='step', monitor_metrics=monitor_metrics)

    def run_on_epoch_end_hook(self, model):
        with self.profiler.profile('on_epoch_end'):
            # callbacks
            self.on_epoch_end()
            # model hooks
            if self.is_function_implemented('on_epoch_end'):
                model.on_epoch_end()

        with self.profiler.profile('on_train_epoch_end'):
            # callbacks
            self.on_train_epoch_end()

            # model hooks
            if self.is_function_implemented('on_train_epoch_end'):
                model.on_train_epoch_end()

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
            self.log_metrics(epoch_log_metrics, {})

        # add metrics to callbacks
        self.callback_metrics.update(epoch_callback_metrics)

        # add metrics to progress_bar
        if len(epoch_progress_bar_metrics) > 0:
            self.add_progress_bar_metrics(epoch_progress_bar_metrics)

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

    def sync_horovod(self):
        if self.use_horovod:
            hvd.join(hvd.local_rank() if self.on_gpu else -1)

    def increment_accumulated_grad_global_step(self):
        # progress global step according to grads progress
        if ((self.batch_idx + 1) % self.accumulate_grad_batches == 0
                or (self.batch_idx + 1) == self.num_training_batches):
            self.global_step += 1
        self.total_batch_idx += 1

    def save_train_loop_metrics_to_loggers(self, batch_idx, batch_output):
        # when metrics should be logged
        should_log_metrics = (batch_idx + 1) % self.row_log_interval == 0 or self.should_stop
        if should_log_metrics or self.fast_dev_run:
            # logs user requested information to logger
            metrics = batch_output.batch_log_metrics
            grad_norm_dic = batch_output.grad_norm_dic
            if len(metrics) > 0 or len(grad_norm_dic) > 0:
                self.log_metrics(metrics, grad_norm_dic)

    def save_loggers_in_training_loop(self, batch_idx):
        # when loggers should save to disk
        should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or self.should_stop
        if should_save_log or self.fast_dev_run:
            if self.is_global_zero and self.logger is not None:
                self.logger.save()

    def should_check_val(self, batch_idx, is_last_batch):
        # decide if we should run validation
        is_val_check_batch = (batch_idx + 1) % self.val_check_batch == 0
        can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
        can_check_val = self.enable_validation and can_check_epoch
        should_check_val = is_val_check_batch or self.should_stop
        is_last_batch_for_infinite_dataset = (is_last_batch and self.val_check_batch == float('inf'))
        should_check_val = can_check_val and (should_check_val or is_last_batch_for_infinite_dataset)

        return should_check_val

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
                opt_closure_result = self.training_step_and_backward(
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
                    train_step_and_backward_closure = lambda: self.training_step_and_backward(
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
            self.callback_metrics.update({k: v for d in batch_callback_metrics for k, v in d.items()})

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            batch_log_metrics=batch_log_metrics,
            training_step_output_for_epoch_end=batch_outputs
        )
        return result

    def training_step_and_backward(self, split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        # lightning module hook
        result = self.train_loop.training_step(split_batch, batch_idx, opt_idx, hiddens)

        # backward pass
        self.train_loop.backward(result, optimizer, opt_idx)

        # hook
        self.train_loop.on_after_backward(result.training_step_output, batch_idx, result.loss)

        return result

    def build_train_args(self, batch, batch_idx, opt_idx, hiddens):
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        if len(self.optimizers) > 1:
            if self.has_arg('training_step', 'optimizer_idx'):
                args.append(opt_idx)
            else:
                num_opts = len(self.optimizers)
                raise ValueError(
                    f'Your LightningModule defines {num_opts} optimizers but '
                    f'training_step is missing the "optimizer_idx" argument.'
                )

        # pass hiddens if using tbptt
        if self.truncated_bptt_steps is not None:
            args.append(hiddens)

        return args

    def update_learning_rates(self, interval: str, monitor_metrics=None):
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            monitor_metrics: dict of possible values to monitor
        """
        if not self.lr_schedulers:
            return

        for scheduler_idx, lr_scheduler in enumerate(self.lr_schedulers):
            current_idx = self.batch_idx if interval == 'step' else self.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency'] == 0:
                # If instance of ReduceLROnPlateau, we need to pass validation loss
                if lr_scheduler['reduce_on_plateau']:
                    monitor_key = lr_scheduler['monitor']

                    if monitor_metrics is not None:
                        monitor_val = monitor_metrics.get(monitor_key)
                    else:
                        monitor_val = self.callback_metrics.get(monitor_key)

                    if monitor_val is None:
                        avail_metrics = ','.join(list(self.callback_metrics.keys()))
                        raise MisconfigurationException(
                            f'ReduceLROnPlateau conditioned on metric {monitor_key}'
                            f' which is not available. Available metrics are: {avail_metrics}.'
                            ' Condition can be set using `monitor` key in lr scheduler dict'
                        )
                    if self.dev_debugger.enabled:
                        old_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']

                    # update LR
                    lr_scheduler['scheduler'].step(monitor_val)

                    if self.dev_debugger.enabled:
                        new_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']
                        self.dev_debugger.track_lr_schedulers_update(
                            self.batch_idx,
                            interval,
                            scheduler_idx,
                            old_lr,
                            new_lr,
                            monitor_key,
                        )
                else:
                    if self.dev_debugger.enabled:
                        old_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']

                    # update LR
                    lr_scheduler['scheduler'].step()

                    if self.dev_debugger.enabled:
                        new_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']
                        self.dev_debugger.track_lr_schedulers_update(
                            self.batch_idx,
                            interval,
                            scheduler_idx,
                            old_lr, new_lr
                        )
