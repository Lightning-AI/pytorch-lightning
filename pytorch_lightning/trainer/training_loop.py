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

Early stopping
--------------

The trainer already sets up default early stopping for you.
To modify this behavior, pass in your own EarlyStopping callback.

.. code-block:: python

    from pytorch_lightning.callbacks import EarlyStopping

    # DEFAULTS used by Trainer
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    # without passing anything in, uses the default callback above
    trainer = Trainer()

    # pass in your own to override the default callback
    trainer = Trainer(early_stop_callback=early_stop_callback)

    # pass in min_epochs to enable the callback after min_epochs have run
    trainer = Trainer(early_stop_callback=early_stop_callback, min_epochs=5)

    # pass in None to disable it
    trainer = Trainer(early_stop_callback=None)

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


"""

import inspect
from abc import ABC, abstractmethod
import warnings

import numpy as np

from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerTrainLoopMixin(ABC):

    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.max_epochs = None
        self.min_epochs = None
        self.use_ddp = None
        self.use_dp = None
        self.use_ddp2 = None
        self.single_gpu = None
        self.data_parallel_device_ids = None
        self.check_val_every_n_epoch = None
        self.num_training_batches = None
        self.val_check_batch = None
        self.num_val_batches = None
        self.disable_validation = None
        self.fast_dev_run = None
        self.is_iterable_train_dataloader = None
        self.main_progress_bar = None
        self.accumulation_scheduler = None
        self.lr_schedulers = None
        self.enable_early_stop = None
        self.early_stop_callback = None
        self.callback_metrics = None
        self.logger = None
        self.global_step = None
        self.testing = None
        self.log_save_interval = None
        self.proc_rank = None
        self.row_log_interval = None
        self.total_batches = None
        self.truncated_bptt_steps = None
        self.optimizers = None
        self.accumulate_grad_batches = None
        self.use_amp = None
        self.print_nan_grads = None
        self.track_grad_norm = None
        self.model = None
        self.running_loss = None
        self.training_tqdm_dict = None
        self.get_train_dataloader = None
        self.reduce_lr_on_plateau_scheduler = None

    @property
    def max_nb_epochs(self):
        """
        .. warning:: `max_nb_epochs` is deprecated and will be removed in v0.8.0, use `max_epochs` instead.
        """
        warnings.warn("`max_nb_epochs` is deprecated and will be removed in "
                      "v0.8.0, use `max_epochs` instead.", DeprecationWarning)
        return self.max_epochs

    @property
    def min_nb_epochs(self):
        """
        .. warning:: `min_nb_epochs` is deprecated and will be removed in v0.8.0, use `min_epochs` instead.
        """
        warnings.warn("`min_nb_epochs` is deprecated and will be removed in "
                      "v0.8.0, use `min_epochs` instead.", DeprecationWarning)
        return self.min_epochs

    @abstractmethod
    def get_model(self):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def is_function_implemented(self, m):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def run_evaluation(self, test):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def transfer_batch_to_gpu(self, batch, gpu):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def clip_gradients(self):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def print_nan_gradients(self):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def is_overriden(self, m):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def add_tqdm_metrics(self, metrics):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def log_metrics(self, metrics, grad_norm_dic):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def process_output(self, output, train):
        # this is just empty shell for code from other class
        pass

    def train(self):
        model = self.get_model()
        # run all epochs
        for epoch in range(self.current_epoch, self.max_epochs):
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(self.get_train_dataloader().sampler, 'set_epoch'):
                self.get_train_dataloader().sampler.set_epoch(epoch)

            # get model
            model = self.get_model()

            # update training progress in trainer and model
            model.current_epoch = epoch
            self.current_epoch = epoch

            total_val_batches = 0
            if not self.disable_validation:
                # val can be checked multiple times in epoch
                is_val_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
                val_checks_per_epoch = self.num_training_batches // self.val_check_batch
                val_checks_per_epoch = val_checks_per_epoch if is_val_epoch else 0
                total_val_batches = self.num_val_batches * val_checks_per_epoch

            # total batches includes multiple val checks
            self.total_batches = self.num_training_batches + total_val_batches
            self.batch_loss_value = 0  # accumulated grads

            if self.fast_dev_run:
                # limit the number of batches to 2 (1 train and 1 val) in fast_dev_run
                num_iterations = 2
            elif self.is_iterable_train_dataloader:
                # for iterable train loader, the progress bar never ends
                num_iterations = None
            else:
                num_iterations = self.total_batches

            # reset progress bar
            # .reset() doesn't work on disabled progress bar so we should check
            if not self.main_progress_bar.disable:
                self.main_progress_bar.reset(num_iterations)
            desc = f'Epoch {epoch + 1}' if not self.is_iterable_train_dataloader else ''
            self.main_progress_bar.set_description(desc)

            # changing gradient according accumulation_scheduler
            self.accumulation_scheduler.on_epoch_begin(epoch, self)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch=self.current_epoch)
            if self.reduce_lr_on_plateau_scheduler is not None:
                val_loss = self.callback_metrics.get('val_loss')
                if val_loss is None:
                    avail_metrics = ','.join(list(self.callback_metrics.keys()))
                    m = f'ReduceLROnPlateau conditioned on metric val_loss ' \
                        f'which is not available. Available metrics are: {avail_metrics}'
                    raise MisconfigurationException(m)
                self.reduce_lr_on_plateau_scheduler.step(val_loss, epoch=self.current_epoch)

            # early stopping
            met_min_epochs = epoch >= self.min_epochs - 1
            if self.enable_early_stop and (met_min_epochs or self.fast_dev_run):
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch,
                                                                    logs=self.callback_metrics)
                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    self.main_progress_bar.close()
                    return

        self.main_progress_bar.close()

        model.on_train_end()

        if self.logger is not None:
            self.logger.finalize("success")

    def run_training_epoch(self):
        # before epoch hook
        if self.is_function_implemented('on_epoch_start'):
            model = self.get_model()
            model.on_epoch_start()

        # run epoch
        for batch_idx, batch in enumerate(self.get_train_dataloader()):
            # stop epoch if we limited the number of training batches
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx

            model = self.get_model()
            model.global_step = self.global_step

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.run_training_batch(batch, batch_idx)
            batch_result, grad_norm_dic, batch_step_metrics = output

            # when returning -1 from train_step, we end epoch early
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_idx + 1) % self.val_check_batch == 0
            can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            should_check_val = (not self.disable_validation and can_check_epoch and
                                (is_val_check_batch or early_stop_epoch))

            # fast_dev_run always forces val checking after train batch
            if self.fast_dev_run or should_check_val:
                self.run_evaluation(test=self.testing)

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

            self.global_step += 1
            self.total_batch_idx += 1

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if early_stop_epoch or self.fast_dev_run:
                break

        # epoch end hook
        if self.is_function_implemented('on_epoch_end'):
            model = self.get_model()
            model.on_epoch_end()

    def run_training_batch(self, batch, batch_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic, {}

        # hook
        if self.is_function_implemented('on_batch_start'):
            model_ref = self.get_model()
            response = model_ref.on_batch_start(batch)

            if response == -1:
                return -1, grad_norm_dic, {}

        splits = [batch]
        if self.truncated_bptt_steps is not None:
            model_ref = self.get_model()
            splits = model_ref.tbptt_split_batch(batch, self.truncated_bptt_steps)

        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            # call training_step once per optimizer
            for opt_idx, optimizer in enumerate(self.optimizers):
                # make sure only the gradients of the current optimizer's paramaters are calculated 
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                for param in self.get_model().parameters():
                    param.requires_grad = False
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.requires_grad = True

                # wrap the forward step in a closure so second order methods work
                def optimizer_closure():
                    # forward pass
                    output = self.training_forward(
                        split_batch, batch_idx, opt_idx, self.hiddens)

                    closure_loss = output[0]
                    progress_bar_metrics = output[1]
                    log_metrics = output[2]
                    callback_metrics = output[3]
                    self.hiddens = output[4]

                    # accumulate loss
                    # (if accumulate_grad_batches = 1 no effect)
                    closure_loss = closure_loss / self.accumulate_grad_batches

                    # backward pass
                    model_ref = self.get_model()
                    model_ref.backward(self.use_amp, closure_loss, optimizer)

                    # track metrics for callbacks
                    all_callback_metrics.append(callback_metrics)

                    # track progress bar metrics
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)

                    # insert after step hook
                    if self.is_function_implemented('on_after_backward'):
                        model_ref = self.get_model()
                        model_ref.on_after_backward()

                    return closure_loss

                # calculate loss
                loss = optimizer_closure()

                # nan grads
                if self.print_nan_grads:
                    self.print_nan_gradients()

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value += loss.item()

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
                    model.optimizer_step(self.current_epoch, batch_idx,
                                         optimizer, opt_idx, optimizer_closure)

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value)
                    self.batch_loss_value = 0
                    self.avg_loss = np.mean(self.running_loss[-100:])

        # activate batch end hook
        if self.is_function_implemented('on_batch_end'):
            model = self.get_model()
            model.on_batch_end()

        # update progress bar
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics.update({k: v for d in all_callback_metrics for k, v in d.items()})

        return 0, grad_norm_dic, all_log_metrics

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
                raise ValueError(
                    f'Your LightningModule defines {len(self.optimizers)} optimizers but '
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
            batch = self.transfer_batch_to_gpu(batch.copy(), gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_end
        if self.is_overriden('training_end'):
            model_ref = self.get_model()
            output = model_ref.training_end(output)

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output
