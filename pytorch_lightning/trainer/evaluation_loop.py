"""
Validation loop
===============

The lightning validation loop handles everything except the actual computations of your model.
To decide what will happen in your validation loop, define the `validation_step` function.
Below are all the things lightning automates for you in the validation loop.

.. note:: Lightning will run 5 steps of validation in the beginning of training as a sanity
 check so you don't have to wait until a full epoch to catch possible validation issues.

Check validation every n epochs
-------------------------------

If you have a small dataset you might want to check validation every n epochs

.. code-block:: python

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)

Set how much of the validation set to check
-------------------------------------------

If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag

val_percent_check will be overwritten by overfit_pct if `overfit_pct > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_percent_check=1.0)

    # check 10% only
    trainer = Trainer(val_percent_check=0.1)

Set how much of the test set to check
-------------------------------------

If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag

test_percent_check will be overwritten by overfit_pct if `overfit_pct > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(test_percent_check=1.0)

    # check 10% only
    trainer = Trainer(test_percent_check=0.1)

Set validation check frequency within 1 training epoch
------------------------------------------------------

For large datasets it's often desirable to check validation multiple times within a training loop.
 Pass in a float to check that often within 1 training epoch.
 Pass in an int k to check every k training batches. Must use an int if using an IterableDataset.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for IterableDatasets or fixed frequency)
    trainer = Trainer(val_check_interval=100)


Set the number of validation sanity steps
-----------------------------------------

Lightning runs a few steps of validation in the beginning of training.
 This avoids crashing in the validation loop sometime deep into a lengthy training loop.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(num_sanity_val_steps=5)


You can use `Trainer(num_sanity_val_steps=0)` to skip the sanity check.

# Testing loop

To ensure you don't accidentally use test data to guide training decisions Lightning
 makes running the test set deliberate.

**test**

You have two options to run the test set.
First case is where you test right after a full training routine.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # run test set
    trainer.test()


Second case is where you load a model and run the test set

.. code-block:: python

    model = MyLightningModule.load_from_metrics(
        weights_path='/path/to/pytorch_checkpoint.ckpt',
        tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
        on_gpu=True,
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this second case, the options you pass to trainer will be used when running
 the test set (ie: 16-bit, dp, ddp, etc...)

"""

import sys
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_warn

try:
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class TrainerEvaluationLoopMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    test_progress_bar: ...
    val_progress_bar: ...
    main_progress_bar: ...
    use_ddp: bool
    use_dp: bool
    use_ddp2: bool
    single_gpu: bool
    data_parallel_device_ids: ...
    model: LightningModule
    num_test_batches: int
    num_val_batches: int
    fast_dev_run: ...
    process_position: ...
    process_output: ...
    training_tqdm_dict: ...
    proc_rank: int
    current_epoch: int
    callback_metrics: ...
    test_dataloaders: DataLoader
    val_dataloaders: DataLoader
    use_tpu: bool
    reload_dataloaders_every_epoch: ...
    progress_bar_refresh_rate: ...

    # Callback system
    on_validation_start: Callable
    on_validation_end: Callable
    on_test_start: Callable
    on_test_end: Callable

    @abstractmethod
    def copy_trainer_model_properties(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def get_model(self):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_overriden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_tpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_gpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_tqdm_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def log_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_test_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_val_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def _evaluate(self, model: LightningModule, dataloaders, max_batches: int, test_mode: bool = False):
        """Run evaluation code.

        Args:
            model: PT model
            dataloaders: list of PT dataloaders
            max_batches: Scalar
            test_mode:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run validation
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []

            # on TPU we have to wrap it under the ParallelLoader
            if self.use_tpu:
                device = xm.xla_device()
                dataloader = xla_pl.ParallelLoader(dataloader, [device])
                dataloader = dataloader.per_device_loader(device)

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)

                # on dp / ddp2 might still want to do something with the batch parts
                if test_mode:
                    if self.is_overriden('test_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('test_step_end'):
                            output = model_ref.test_step_end(output)
                else:
                    if self.is_overriden('validation_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('validation_step_end'):
                            output = model_ref.validation_step_end(output)

                # track outputs for collation
                dl_outputs.append(output)

                # batch done
                if self.progress_bar_refresh_rate >= 1 and batch_idx % self.progress_bar_refresh_rate == 0:
                    if test_mode:
                        self.test_progress_bar.update(self.progress_bar_refresh_rate)
                    else:
                        self.val_progress_bar.update(self.progress_bar_refresh_rate)
                        self.main_progress_bar.update(self.progress_bar_refresh_rate)
            outputs.append(dl_outputs)

        eval_results = {}

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        if isinstance(model, (LightningDistributedDataParallel, LightningDataParallel)):
            model = model.module

        if test_mode:
            if self.is_overriden('test_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.test_end(outputs)
                rank_zero_warn('Method `test_end` was deprecated in 0.7.0 and will be removed 1.0.0.'
                               ' Use `test_epoch_end` instead.', DeprecationWarning)

            elif self.is_overriden('test_epoch_end', model=model):
                eval_results = model.test_epoch_end(outputs)

        else:
            if self.is_overriden('validation_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.validation_end(outputs)
                rank_zero_warn('Method `validation_end` was deprecated in 0.7.0 and will be removed 1.0.0.'
                               ' Use `validation_epoch_end` instead.', DeprecationWarning)

            elif self.is_overriden('validation_epoch_end', model=model):
                eval_results = model.validation_epoch_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def run_evaluation(self, test_mode: bool = False):
        # when testing make sure user defined a test step
        if test_mode and not self.is_overriden('test_step'):
            raise MisconfigurationException(
                "You called `.test()` without defining model's `.test_step()`."
                " Please define and try again")

        # Validation/Test begin callbacks
        if test_mode:
            self.on_test_start()
        else:
            self.on_validation_start()

        # hook
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test_mode:
            if self.test_dataloaders is None:
                self.reset_test_dataloader(model)

            dataloaders = self.test_dataloaders
            max_batches = self.num_test_batches
        else:
            # val
            if self.val_dataloaders is None:
                self.reset_val_dataloader(model)

            dataloaders = self.val_dataloaders
            max_batches = self.num_val_batches

        # cap max batches to 1 when using fast_dev_run
        if self.fast_dev_run:
            max_batches = 1

        # init validation or test progress bar
        # main progress bar will already be closed when testing so initial position is free
        position = 2 * self.process_position + (not test_mode)
        desc = 'Testing' if test_mode else 'Validating'
        total = max_batches if max_batches != float('inf') else None
        pbar = tqdm(desc=desc, total=total, leave=test_mode, position=position,
                    disable=not self.progress_bar_refresh_rate, dynamic_ncols=True, file=sys.stdout)
        setattr(self, f'{"test" if test_mode else "val"}_progress_bar', pbar)

        # run evaluation
        eval_results = self._evaluate(self.model, dataloaders, max_batches, test_mode)
        _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(eval_results)

        # add metrics to prog bar
        self.add_tqdm_metrics(prog_bar_metrics)

        # log results of test
        if test_mode and self.proc_rank == 0:
            print('-' * 80)
            print('TEST RESULTS')
            pprint(callback_metrics)
            print('-' * 80)

        # log metrics
        self.log_metrics(log_metrics, {})

        # track metrics for callbacks
        self.callback_metrics.update(callback_metrics)

        # hook
        model.on_post_performance_check()

        # add model specific metrics
        if not test_mode:
            self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # close progress bar
        if test_mode:
            self.test_progress_bar.close()
        else:
            self.val_progress_bar.close()

        # eventual dataset reloading
        if test_mode:
            if self.reload_dataloaders_every_epoch:
                self.reset_test_dataloader(model)
        else:
            # val
            if self.reload_dataloaders_every_epoch:
                self.reset_val_dataloader(model)

        # Validation/Test end callbacks
        if test_mode:
            self.on_test_end()

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test_mode: bool = False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test_mode and len(self.test_dataloaders) > 1:
            args.append(dataloader_idx)

        elif not test_mode and len(self.val_dataloaders) > 1:
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # single GPU data transfer
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # TPU data  transfer
        if self.use_tpu:
            batch = self.transfer_batch_to_tpu(batch)
            args[0] = batch

        # CPU, TPU or gpu step
        if test_mode:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output
