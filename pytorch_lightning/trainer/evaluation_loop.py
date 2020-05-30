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

    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path='/path/to/pytorch_checkpoint.ckpt',
        hparams_file='/path/to/test_tube/experiment/version/hparams.yaml',
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this second case, the options you pass to trainer will be used when running
 the test set (ie: 16-bit, dp, ddp, etc...)

"""

from abc import ABC, abstractmethod
from pprint import pprint
from typing import Callable

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.core.step_result import EvalResult, Result

try:
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except ImportError:
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class TrainerEvaluationLoopMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    use_ddp: bool
    use_dp: bool
    use_ddp2: bool
    use_horovod: bool
    single_gpu: bool
    data_parallel_device_ids: ...
    model: LightningModule
    num_test_batches: int
    num_val_batches: int
    fast_dev_run: ...
    progress_bar_dict: ...
    proc_rank: int
    current_epoch: int
    callback_metrics: ...
    test_dataloaders: DataLoader
    val_dataloaders: DataLoader
    use_tpu: bool
    reload_dataloaders_every_epoch: ...
    tpu_id: int

    # Callback system
    on_validation_batch_start: Callable
    on_validation_batch_end: Callable
    on_test_batch_start: Callable
    on_test_batch_end: Callable
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
    def is_overridden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_tpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_gpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_progress_bar_metrics(self, *args):
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
        """
        Runs full evaluation (test or validation) on all the dataloaders

        Args:
            model: PT model
            dataloaders: list of PT dataloaders
            max_batches: Scalar
            test_mode:
        """
        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # ----------------------------------
        # disable grads BN, DO ... for eval
        # ----------------------------------
        model.zero_grad()
        model.eval()
        torch.set_grad_enabled(False)

        # ----------------------------------
        # validation for each dataloader
        # ----------------------------------
        all_dataloader_outputs = []
        for dataloader_idx, dataloader in enumerate(dataloaders):

            # on TPU we have to wrap it under the ParallelLoader
            if self.use_tpu:
                device = xm.xla_device(self.tpu_id)
                dataloader = xla_pl.ParallelLoader(dataloader, [device])
                dataloader = dataloader.per_device_loader(device)

            # ----------------------------------
            # run a loop through each dataloader
            # ----------------------------------
            dataloader_outputs = []
            for batch_idx, batch in enumerate(dataloader):
                # ignore null batches
                if batch is None:
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # run the dataloader step
                # eval_step_result is an EvalResult object
                eval_step_result = self._dataloader_eval_step(model, batch, batch_idx, dataloader_idx, test_mode)
                dataloader_outputs.append(eval_step_result)

            # ----------------------------------
            # track to merge all the dataloader outputs
            # ----------------------------------
            all_dataloader_outputs.append(dataloader_outputs)

        # -----------------------
        # format epoch_end inputs
        # -----------------------
        # only use .to_epoch_end dict
        epoch_end_inputs = []
        for dl_output_list in all_dataloader_outputs:
            to_epoch_ends = [x.to_epoch_end for x in dl_output_list]
            epoch_end_inputs.append(to_epoch_ends)

        # with a single dataloader don't pass an array of dataloader outputs
        if len(dataloaders) == 1:
            epoch_end_inputs = epoch_end_inputs[0]

        # get model from parallel wrapper
        model_ref = self.get_model()

        # -----------------------------
        # RUN XXX_EPOCH_END
        # -----------------------------
        # eval_epoch_end_result = EvalResult()
        eval_key = 'test' if test_mode else 'validation'
        if self.is_overridden(f'{eval_key}_end', model=model_ref):
            # TODO: remove in v1.0.0
            test_end_fx = getattr(model, f'{eval_key}_end')
            eval_epoch_end_result = test_end_fx(epoch_end_inputs)
            rank_zero_warn(f'Method `{eval_key}_end` was deprecated in v0.7 and will be removed v1.0.'
                           f' Use `{eval_key}_epoch_end` instead.', DeprecationWarning)

        elif self.is_overridden(f'{eval_key}_epoch_end', model=model_ref):
            test_epoch_end_fx = getattr(model, f'{eval_key}_epoch_end')
            eval_epoch_end_result = test_epoch_end_fx(epoch_end_inputs)

        # TODO: apply key reductions here if test_epoch_end was not used

        # TODO: figure out eval_epoch_end_result
        # -------------------------------------
        # MAP SIMPLE DICT TO STRUCTURED RESULT
        # -------------------------------------
        if not isinstance(eval_epoch_end_result, EvalResult):
            assert isinstance(eval_epoch_end_result, dict), f'output of {eval_key}_epoch_end must be dict or EvalResult'
            result = EvalResult()

            # TODO: pull key from callbacks
            callback_key = 'val_loss'
            result.checkpoint_on = eval_epoch_end_result.get(callback_key)
            result.early_stop_on = eval_epoch_end_result.get(callback_key)

            result.log_on_epoch_end = eval_epoch_end_result.get('log')
            result.pbar_on_epoch_end = eval_epoch_end_result.get('progress_bar')

        # -----------------------
        # enable training mode
        # -----------------------
        model.train()
        torch.set_grad_enabled(True)

        return eval_epoch_end_result

    def _dataloader_eval_step(self, model, batch, batch_idx, dataloader_idx, test_mode) -> EvalResult:
        """
        Runs through the following sequence
        - on_xxx_batch_start
        - XXX_step
        - XXX_step_end
        - XXX_epoch_end
        - on_xxx_batch_end

        Args:
            model:
            batch:
            batch_idx:
            dataloader_idx:
            test_mode:

        Returns: EvalResult
        """
        # -------------------------------------
        # ON_XXX_BATCH_START CALLBACK
        # -------------------------------------
        if test_mode:
            self.on_test_batch_start()
        else:
            self.on_validation_batch_start()

        # -------------------------------------
        # VALIDATION_STEP OR TEST_STEP
        # -------------------------------------
        if self.use_amp and self.use_native_amp:
            with torch.cuda.amp.autocast():
                eval_step_output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)
        else:
            eval_step_output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)

        # init the eval step result for this dataloader
        eval_step_result = EvalResult()
        if isinstance(eval_step_output, EvalResult):
            eval_step_result = eval_step_output

        # -------------------------------------
        # VALIDATION_STEP_END OR TEST_STEP_END
        # -------------------------------------
        # on dp / ddp2 might still want to do something with the batch parts
        # the result of this step will also be sent to on_epoch_end
        callback_name = 'test_step_end' if test_mode else 'validation_step_end'
        if self.is_overridden(callback_name):

            # -------------------------------------
            # map simple dict to what batch end needs
            # -------------------------------------
            if not isinstance(eval_step_output, EvalResult):
                eval_step_result.to_batch_end = eval_step_output

            # TODO: add warning if user overrode this method and did not pass in a `to_xxx_step_end` key

            # get the model within parallel wrapper
            model_ref = self.get_model()

            # ------------------------
            # XXX_STEP_END
            # ------------------------
            with self.profiler.profile(callback_name):
                callback_fx = getattr(model_ref, callback_name)
                batch_step_end_output = callback_fx(eval_step_output.to_batch_end)

            # if step_end returned a Result use this as the new output
            if isinstance(batch_step_end_output, EvalResult):
                eval_step_result = batch_step_end_output

            # dict result, pass to epoch end in that case
            else:
                eval_step_result.to_epoch_end = batch_step_end_output

        # -------------------------------------
        # ON_XXX_BATCH_END CALLBACK
        # -------------------------------------
        # call the `on_test_batch_end` or `on_validation_batch_end` function
        on_batch_end_fx = self.on_test_batch_end if test_mode else self.on_validation_batch_end
        on_batch_end_fx()

        return eval_step_result

    def run_evaluation(self, test_mode: bool = False):
        # hook
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test_mode:
            self.reset_test_dataloader(model)

            dataloaders = self.test_dataloaders
            max_batches = self.num_test_batches
        else:
            # val
            if self.val_dataloaders is None:
                self.reset_val_dataloader(model)

            dataloaders = self.val_dataloaders
            max_batches = self.num_val_batches

        # enable fast_dev_run without val loop
        if dataloaders is None:
            return

        # cap max batches to 1 when using fast_dev_run
        if self.fast_dev_run:
            max_batches = 1

        # Validation/Test begin callbacks
        if test_mode:
            self.on_test_start()
        else:
            self.on_validation_start()

        # run evaluation
        eval_results = self._evaluate(self.model, dataloaders, max_batches, test_mode)

        if isinstance(eval_results, Result):
            eval_results = self.process_step_result(eval_results)
        else:
            eval_results = self.process_output(eval_results)

        # add metrics to prog bar
        self.add_progress_bar_metrics(eval_results.pbar_on_epoch_end)

        # log results of test
        if test_mode and self.proc_rank == 0:
            print('-' * 80)
            print('TEST RESULTS')
            pprint(eval_results.log_on_epoch_end)
            print('-' * 80)

        # log metrics
        self.log_metrics(eval_results.log_on_epoch_end, {})

        # track metrics for callbacks
        self.callback_metrics.update(eval_results.callback_metrics)

        # hook
        model.on_post_performance_check()

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
        else:
            self.on_validation_end()

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test_mode: bool = False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if (test_mode and len(self.test_dataloaders) > 1) \
                or (not test_mode and len(self.val_dataloaders) > 1):
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # Horovod
        if self.use_horovod and self.on_gpu:
            batch = self.transfer_batch_to_gpu(batch, hvd.local_rank())
            args[0] = batch

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
            batch = self.transfer_batch_to_tpu(batch, self.tpu_id)
            args[0] = batch

        # CPU, TPU or gpu step
        if test_mode:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)
        return output
