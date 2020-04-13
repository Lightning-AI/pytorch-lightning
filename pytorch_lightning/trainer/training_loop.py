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

train_percent_check will be overwritten by overfit_pct if `overfit_pct > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(train_percent_check=1.0)

    # check 10% only
    trainer = Trainer(train_percent_check=0.1)

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

import copy
from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, List

import numpy as np
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import rank_zero_warn

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class TrainerTrainLoopMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    max_epochs: int
    min_epochs: int
    use_ddp: bool
    use_dp: bool
    use_ddp2: bool
    single_gpu: bool
    use_tpu: bool
    data_parallel_device_ids: ...
    check_val_every_n_epoch: ...
    num_training_batches: int
    val_check_batch: ...
    num_val_batches: int
    disable_validation: bool
    fast_dev_run: ...
    main_progress_bar: ...
    accumulation_scheduler: ...
    lr_schedulers: ...
    enable_early_stop: ...
    early_stop_callback: ...
    callback_metrics: ...
    logger: Union[LightningLoggerBase, bool]
    global_step: int
    testing: bool
    log_save_interval: float
    proc_rank: int
    row_log_interval: float
    total_batches: int
    truncated_bptt_steps: ...
    optimizers: ...
    optimizer_frequencies: ...
    accumulate_grad_batches: int
    track_grad_norm: ...
    model: LightningModule
    interrupted: bool
    running_loss: ...
    training_tqdm_dict: ...
    reduce_lr_on_plateau_scheduler: ...
    profiler: ...
    batch_idx: int
    precision: ...
    train_dataloader: DataLoader
    reload_dataloaders_every_epoch: bool
    progress_bar_refresh_rate: ...
    max_steps: int
    min_steps: int
    total_batch_idx: int
    checkpoint_callback: ...
    terminate_on_nan: bool

    # Callback system
    callbacks: List[Callback]
    on_train_start: Callable
    on_train_end: Callable
    on_batch_start: Callable
    on_batch_end: Callable
    on_epoch_start: Callable
    on_epoch_end: Callable
    on_validation_end: Callable

    @abstractmethod
    def get_model(self):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_function_implemented(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def run_evaluation(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_gpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_tpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def clip_gradients(self):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def detect_nan_tensors(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_overriden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_tqdm_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def log_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def process_output(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_train_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_val_dataloader(self, model):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def has_arg(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def train(self):
        rank_zero_warn('Displayed epoch numbers in the progress bar start from "1" until v0.6.x,'
                       ' but will start from "0" in v0.8.0.', RuntimeWarning)

        # get model
        model = self.get_model()

        # load data
        # if reload_dataloaders_every_epoch, this is moved to the epoch loop
        if not self.reload_dataloaders_every_epoch:
            self.reset_train_dataloader(model)
        self.reset_val_dataloader(model)

        # Train start events
        with self.profiler.profile('on_train_start'):
            # callbacks
            self.on_train_start()
            # initialize early stop callback
            if self.early_stop_callback is not None:
                self.early_stop_callback.on_train_start(self, self.get_model())
            # model hooks
            model.on_train_start()

        try:
            # run all epochs
            for epoch in range(self.current_epoch, self.max_epochs):
                # reset train dataloader
                if self.reload_dataloaders_every_epoch:
                    self.reset_train_dataloader(model)
                # set seed for distributed sampler (enables shuffling for each epoch)
                if self.use_ddp \
                        and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                # update training progress in trainer and model
                model.current_epoch = epoch
                self.current_epoch = epoch

                total_val_batches = 0
                is_val_epoch = False
                if not self.disable_validation and self.num_training_batches != float('inf'):
                    # val can be checked multiple times in epoch
                    is_val_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
                    val_checks_per_epoch = self.num_training_batches // self.val_check_batch
                    val_checks_per_epoch = val_checks_per_epoch if is_val_epoch else 0
                    total_val_batches = self.num_val_batches * val_checks_per_epoch

                # total batches includes multiple val checks
                self.total_batches = self.num_training_batches + total_val_batches

                # changing gradient according accumulation_scheduler
                self.accumulation_scheduler.on_epoch_start(self, self.get_model())

                # stores accumulated grad fractions per batch
                self.batch_loss_value = TensorRunningAccum(
                    window_length=self.accumulate_grad_batches
                )

                if self.fast_dev_run:
                    # limit the number of batches to 2 (1 train and 1 val) in fast_dev_run
                    num_iterations = 2
                elif self.total_batches == float('inf'):
                    # for infinite train or val loader, the progress bar never ends
                    num_iterations = None
                else:
                    num_iterations = self.total_batches

                # reset progress bar
                # .reset() doesn't work on disabled progress bar so we should check
                if not self.main_progress_bar.disable:
                    self.main_progress_bar.reset(num_iterations)
                desc = f'Epoch {epoch + 1}'
                self.main_progress_bar.set_description(desc)

                # -----------------
                # RUN TNG EPOCH
                # -----------------
                self.run_training_epoch()

                # update LR schedulers
                self.update_learning_rates(interval='epoch')

                if self.max_steps and self.max_steps == self.global_step:
                    self.run_training_teardown()
                    return

                # early stopping
                met_min_epochs = epoch >= self.min_epochs - 1
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                # TODO wrap this logic into the callback
                if self.enable_early_stop:
                    if (met_min_epochs and met_min_steps) or self.fast_dev_run:
                        should_stop = self.early_stop_callback.on_epoch_end(self, self.get_model())
                        # stop training
                        stop = should_stop and met_min_epochs
                        if stop:
                            self.run_training_teardown()
                            return

            self.run_training_teardown()

        except KeyboardInterrupt:
            if self.proc_rank == 0:
                log.info('Detected KeyboardInterrupt, attempting graceful shutdown...')
            self.interrupted = True
            self.run_training_teardown()

    def run_training_epoch(self):

        # get model
        model = self.get_model()

        # Epoch start events
        with self.profiler.profile('on_epoch_start'):
            # callbacks
            self.on_epoch_start()

            # model hooks
            if self.is_function_implemented('on_epoch_start'):
                model.on_epoch_start()

        # track local dataloader so TPU can wrap each epoch
        train_dataloader = self.train_dataloader

        # on TPU we have to wrap it under the ParallelLoader
        if self.use_tpu:
            device = xm.xla_device()
            train_dataloader = xla_pl.ParallelLoader(train_dataloader, [device])
            train_dataloader = train_dataloader.per_device_loader(device)

        # bookkeeping
        outputs = []

        # run epoch
        for batch_idx, (batch, is_last_batch) in self.profiler.profile_iterable(
            enumerate(_with_is_last(train_dataloader)), "get_train_batch"
        ):
            # stop epoch if we limited the number of training batches
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx

            model.global_step = self.global_step

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            _outputs = self.run_training_batch(batch, batch_idx)
            batch_result, grad_norm_dic, batch_step_metrics, batch_output = _outputs
            # detach tensors in batch_output before appending to outputs
            outputs.append(_recursive_detach(batch_output))

            # when returning -1 from train_step, we end epoch early
            early_stop_epoch = batch_result == -1

            # update lr
            self.update_learning_rates(interval='step')

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_idx + 1) % self.val_check_batch == 0
            can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            can_check_val = not self.disable_validation and can_check_epoch
            should_check_val = is_val_check_batch or early_stop_epoch
            should_check_val = should_check_val or (is_last_batch and self.val_check_batch == float('inf'))
            should_check_val = can_check_val and should_check_val

            # fast_dev_run always forces val checking after train batch
            if self.fast_dev_run or should_check_val:
                self.run_evaluation(test_mode=self.testing)

            # when logs should be saved
            should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log or self.fast_dev_run:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            should_log_metrics = batch_idx % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics or self.fast_dev_run:
                # logs user requested information to logger
                self.log_metrics(batch_step_metrics, grad_norm_dic)

            # ---------------
            # CHECKPOINTING, EARLY STOPPING
            # ---------------
            # save checkpoint even when no test or val step are defined
            if self.fast_dev_run or should_check_val:
                self.call_checkpoint_callback()

                if self.enable_early_stop:
                    self.early_stop_callback.check_metrics(self.callback_metrics)

            # progress global step according to grads progress
            if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:
                self.global_step += 1
            self.total_batch_idx += 1

            # max steps reached, end training
            if self.max_steps is not None and self.max_steps == self.global_step:
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if early_stop_epoch or self.fast_dev_run:
                break

        # process epoch outputs
        if isinstance(model, (LightningDistributedDataParallel, LightningDataParallel)):
            model = model.module

        if self.is_overriden('training_epoch_end', model=model):
            epoch_output = model.training_epoch_end(outputs)
            _processed_outputs = self.process_output(epoch_output)
            log_epoch_metrics = _processed_outputs[2]
            callback_epoch_metrics = _processed_outputs[3]
            self.log_metrics(log_epoch_metrics, {})
            self.callback_metrics.update(callback_epoch_metrics)

        # in case validation step is missing and you are not running fast-dev to duplicate last batch
        if not self.is_overriden('validation_step') and not (self.fast_dev_run or should_check_val):
            self.call_checkpoint_callback()

            if self.enable_early_stop:
                self.early_stop_callback.check_metrics(self.callback_metrics)

        # Epoch end events
        with self.profiler.profile('on_epoch_end'):
            # callbacks
            self.on_epoch_end()
            # model hooks
            if self.is_function_implemented('on_epoch_end'):
                model.on_epoch_end()

    def run_training_batch(self, batch, batch_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic, {}, {}

        # Batch start events
        with self.profiler.profile('on_batch_start'):
            # callbacks
            self.on_batch_start()
            # hooks
            if self.is_function_implemented('on_batch_start'):
                response = self.get_model().on_batch_start(batch)
                if response == -1:
                    return -1, grad_norm_dic, {}, {}

        splits = [batch]
        if self.truncated_bptt_steps is not None:
            model_ref = self.get_model()
            with self.profiler.profile('tbptt_split_batch'):
                splits = model_ref.tbptt_split_batch(batch, self.truncated_bptt_steps)

        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            for opt_idx, optimizer in self._get_optimizers_iterable():
                # make sure only the gradients of the current optimizer's paramaters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                # wrap the forward step in a closure so second order methods work
                def optimizer_closure():
                    # forward pass
                    with self.profiler.profile('model_forward'):
                        output_dict = self.training_forward(
                            split_batch, batch_idx, opt_idx, self.hiddens)

                        # format and reduce outputs accordingly
                        processed_output = self.process_output(output_dict, train=True)

                    closure_loss, progress_bar_metrics, log_metrics, callback_metrics, self.hiddens = processed_output

                    # accumulate loss
                    # (if accumulate_grad_batches = 1 no effect)
                    closure_loss = closure_loss / self.accumulate_grad_batches

                    # backward pass
                    model_ref = self.get_model()
                    with self.profiler.profile('model_backward'):
                        model_ref.backward(self, closure_loss, optimizer, opt_idx)

                    # track metrics for callbacks
                    all_callback_metrics.append(callback_metrics)

                    # track progress bar metrics
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)

                    # insert after step hook
                    if self.is_function_implemented('on_after_backward'):
                        model_ref = self.get_model()
                        with self.profiler.profile('on_after_backward'):
                            model_ref.on_after_backward()

                    return closure_loss, output_dict

                # calculate loss
                loss, batch_output = optimizer_closure()

                # check if loss or model weights are nan
                if self.terminate_on_nan:
                    self.detect_nan_tensors(loss)

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value.append(loss)

                # gradient update with accumulated gradients
                if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:

                    # track gradient norms when requested
                    if batch_idx % self.row_log_interval == 0:
                        if self.track_grad_norm > 0:
                            model = self.get_model()
                            grad_norm_dic = model.grad_norm(
                                self.track_grad_norm)

                    # clip gradients
                    self.clip_gradients()

                    # calls .step(), .zero_grad()
                    # override function to modify this behavior
                    model = self.get_model()
                    with self.profiler.profile('optimizer_step'):
                        model.optimizer_step(self.current_epoch, batch_idx,
                                             optimizer, opt_idx,
                                             lambda: optimizer_closure()[0])

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value.mean())

                    # reset for next set of accumulated grads
                    self.batch_loss_value.reset()

        # Batch end events
        with self.profiler.profile('on_batch_end'):
            # callbacks
            self.on_batch_end()
            # model hooks
            if self.is_function_implemented('on_batch_end'):
                self.get_model().on_batch_end()

        # update progress bar
        if self.progress_bar_refresh_rate >= 1 and batch_idx % self.progress_bar_refresh_rate == 0:
            self.main_progress_bar.update(self.progress_bar_refresh_rate)
            self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics.update({k: v for d in all_callback_metrics for k, v in d.items()})

        return 0, grad_norm_dic, all_log_metrics, batch_output

    def _get_optimizers_iterable(self):
        if not self.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.optimizers))

        optimizer_freq_cumsum = np.cumsum(self.optimizer_frequencies)
        optimizers_loop_length = optimizer_freq_cumsum[-1]
        current_place_in_loop = self.total_batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = np.argmax(optimizer_freq_cumsum > current_place_in_loop)
        return [(opt_idx, self.optimizers[opt_idx])]

    def run_training_teardown(self):
        self.main_progress_bar.close()

        # Train end events
        with self.profiler.profile('on_train_end'):
            # callbacks
            self.on_train_end()
            # model hooks
            if self.is_function_implemented('on_train_end'):
                self.get_model().on_train_end()

        if self.logger is not None:
            self.logger.finalize("success")

        # summarize profile results
        self.profiler.describe()

    def training_forward(self, batch, batch_idx, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_idx:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
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

        # distributed forward
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            output = self.model(*args)

        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if isinstance(self.data_parallel_device_ids, list):
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(copy.copy(batch), gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # TPU support
        elif self.use_tpu:
            batch = self.transfer_batch_to_tpu(copy.copy(batch))
            args[0] = batch
            output = self.model.training_step(*args)

        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_step_end
        # do something will all the dp outputs (like softmax)
        if self.is_overriden('training_step_end'):
            model_ref = self.get_model()
            with self.profiler.profile('training_step_end'):
                output = model_ref.training_step_end(output)

        # allow any mode to define training_end
        # TODO: remove in 1.0.0
        if self.is_overriden('training_end'):
            model_ref = self.get_model()
            with self.profiler.profile('training_end'):
                output = model_ref.training_end(output)

            rank_zero_warn('`training_end` was deprecated in 0.7.0 and will be removed 1.0.0.'
                           ' Use training_epoch_end instead', DeprecationWarning)

        return output

    def update_learning_rates(self, interval: str):
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
        """
        if not self.lr_schedulers:
            return

        for lr_scheduler in self.lr_schedulers:
            current_idx = self.batch_idx if interval == 'step' else self.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency'] == 0:
                # If instance of ReduceLROnPlateau, we need to pass validation loss
                if lr_scheduler['reduce_on_plateau']:
                    monitor_key = lr_scheduler['monitor']
                    monitor_val = self.callback_metrics.get(monitor_key)
                    if monitor_val is None:
                        avail_metrics = ','.join(list(self.callback_metrics.keys()))
                        raise MisconfigurationException(
                            f'ReduceLROnPlateau conditioned on metric {monitor_key}'
                            f' which is not available. Available metrics are: {avail_metrics}.'
                            ' Condition can be set using `monitor` key in lr scheduler dict'
                        )
                    lr_scheduler['scheduler'].step(monitor_val)
                else:
                    lr_scheduler['scheduler'].step()

    def call_checkpoint_callback(self):
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.on_validation_end(self, self.get_model())
        self.on_validation_end()


def _with_is_last(iterable):
    """Pass through values from the given iterable with an added boolean indicating if this is the last item.
    See `https://stackoverflow.com/a/1630350 <https://stackoverflow.com/a/1630350>`_"""
    it = iter(iterable)
    last = next(it)
    for val in it:
        # yield last and has next
        yield last, False
        last = val
    # yield last, no longer has next
    yield last, True


def _recursive_detach(in_dict):
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Parameters
    ----------
    in_dict : dict

    Returns
    -------
    out_dict : dict
    """
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            out_dict.update({k: _recursive_detach(v)})
        elif callable(getattr(v, 'detach', None)):
            out_dict.update({k: v.detach()})
        else:
            out_dict.update({k: v})
    return out_dict
