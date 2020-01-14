"""
# Validation loop

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

from abc import ABC, abstractmethod

import torch
import sys
import tqdm

from pytorch_lightning.utilities.debugging import MisconfigurationException


class TrainerEvaluationLoopMixin(ABC):

    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.test_progress_bar = None
        self.val_progress_bar = None
        self.main_progress_bar = None
        self.use_ddp = None
        self.use_dp = None
        self.use_ddp2 = None
        self.single_gpu = None
        self.data_parallel_device_ids = None
        self.model = None
        self.num_test_batches = None
        self.num_val_batches = None
        self.fast_dev_run = None
        self.process_position = None
        self.show_progress_bar = None
        self.process_output = None
        self.training_tqdm_dict = None
        self.proc_rank = None
        self.checkpoint_callback = None
        self.current_epoch = None
        self.callback_metrics = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None

    @abstractmethod
    def copy_trainer_model_properties(self, model):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def get_model(self):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def is_overriden(self, m):
        # this is just empty shell for code from other class
        pass

    @abstractmethod
    def transfer_batch_to_gpu(self, batch, gpu):
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

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """Run evaluation code.

        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
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

        # run training
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):

                if batch is None:  # pragma: no cover
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.evaluation_forward(model,
                                                 batch,
                                                 batch_idx,
                                                 dataloader_idx,
                                                 test)

                # track outputs for collation
                dl_outputs.append(output)

                # batch done
                if test:
                    self.test_progress_bar.update(1)
                else:
                    self.val_progress_bar.update(1)
                    self.main_progress_bar.update(1)
            outputs.append(dl_outputs)

        eval_results = {}

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        model = self.get_model()
        if test and self.is_overriden('test_end'):
            eval_results = model.test_end(outputs)
        elif self.is_overriden('validation_end'):
            eval_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def run_evaluation(self, test=False):
        # when testing make sure user defined a test step
        if test and not (self.is_overriden('test_step') and self.is_overriden('test_end')):
            m = '''You called `.test()` without defining model's `.test_step()` or `.test_end()`.
                    Please define and try again'''
            raise MisconfigurationException(m)

        # hook
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test:
            dataloaders = self.get_test_dataloaders()
            max_batches = self.num_test_batches
        else:
            # val
            dataloaders = self.get_val_dataloaders()
            max_batches = self.num_val_batches

        # cap max batches to 1 when using fast_dev_run
        if self.fast_dev_run:
            max_batches = 1

        # init validation or test progress bar
        # main progress bar will already be closed when testing so initial position is free
        position = 2 * self.process_position + (not test)
        desc = 'Testing' if test else 'Validating'
        pbar = tqdm.tqdm(desc=desc, total=max_batches, leave=test, position=position,
                         disable=not self.show_progress_bar, dynamic_ncols=True,
                         unit='batch', file=sys.stdout)
        setattr(self, f'{"test" if test else "val"}_progress_bar', pbar)

        # run evaluation
        eval_results = self.evaluate(self.model,
                                     dataloaders,
                                     max_batches,
                                     test)
        _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(
            eval_results)

        # add metrics to prog bar
        self.add_tqdm_metrics(prog_bar_metrics)

        # log metrics
        self.log_metrics(log_metrics, {})

        # track metrics for callbacks
        self.callback_metrics.update(callback_metrics)

        # hook
        model.on_post_performance_check()

        # add model specific metrics
        tqdm_metrics = self.training_tqdm_dict
        if not test:
            self.main_progress_bar.set_postfix(**tqdm_metrics)

        # close progress bar
        if test:
            self.test_progress_bar.close()
        else:
            self.val_progress_bar.close()

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  logs=self.callback_metrics)

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)

        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # CPU
        if test:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output
