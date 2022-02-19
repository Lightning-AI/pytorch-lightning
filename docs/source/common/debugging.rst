.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _debugging:

#########
Debugging
#########

The Lightning :class:`~pytorch_lightning.trainer.trainer.Trainer` is empowered with a lot of flags that can help you debug your :class:`~pytorch_lightning.core.lightning.LightningModule`.

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_debugging.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+7-+debugging_1.mp4"></video>

|

The following are flags that make debugging much easier.


----------------


******************
Quick Unit Testing
******************

fast_dev_run
============

This flag runs a "unit test" by running ``N`` if set to ``N`` (int) else 1 if set to ``True`` training, validation, testing and predict batch(es)
for a single epoch. The point is to have a dry run to detect any bugs in the respective loop without having to wait for a complete loop to crash.

Internally, it just updates ``limit_<train/test/val/predict>_batches=fast_dev_run`` and sets ``max_epoch=1`` to limit the batches.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.fast_dev_run`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # runs 1 train, val, test batch and program ends
    trainer = Trainer(fast_dev_run=True)

    # runs 7 train, val, test batches and program ends
    trainer = Trainer(fast_dev_run=7)

.. note::

    This argument will disable tuner, checkpoint callbacks, early stopping callbacks,
    loggers and logger callbacks like :class:`~pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor` and
    :class:`~pytorch_lightning.callbacks.device_stats_monitor.DeviceStatsMonitor`.


Shorten Epochs
==============

Sometimes it's helpful to only use a fraction of your training, val, test, or predict data (or a set number of batches).
For example, you can use 20% of the training set and 1% of the validation set.

On larger datasets like Imagenet, this can help you debug or test a few things faster than waiting for a full epoch.

.. testcode::

    # use only 10% of training data and 1% of val data
    trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)

    # use 10 batches of train and 5 batches of val
    trainer = Trainer(limit_train_batches=10, limit_val_batches=5)


Validation Sanity Check
=======================

Lightning runs a few steps of validation in the beginning of training.
This avoids crashing in the validation loop sometime deep into a lengthy training loop.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.num_sanity_val_steps`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # DEFAULT
    trainer = Trainer(num_sanity_val_steps=2)


Make Model Overfit on Subset of Data
====================================

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.overfit_batches`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # use only 1% of training data (and turn off validation)
    trainer = Trainer(overfit_batches=0.01)

    # similar, but with a fixed 10 batches
    trainer = Trainer(overfit_batches=10)

When using this flag, validation will be disabled. We will also replace the sampler
in the training set to turn off shuffle for you.


----------------


************
Optimization
************

Inspect Gradient Norms
======================

Logs the norm of the gradients to the logger.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.track_grad_norm`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # the 2-norm
    trainer = Trainer(track_grad_norm=2)


Detect Anomaly
==============

You can enable anomaly detection for the autograd engine. It uses PyTorch's built-in
`Anomaly Detection Context-manager <https://pytorch.org/docs/stable/autograd.html#anomaly-detection>`_.

To enable it within Lightning, use Trainer's flag:

.. testcode::

    trainer = Trainer(detect_anomaly=True)


----------------


***********
Performance
***********

Log Device Statistics
=====================

Monitor and log device stats during training with the :class:`~pytorch_lightning.callbacks.device_stats_monitor.DeviceStatsMonitor`.

.. testcode::

    from pytorch_lightning.callbacks import DeviceStatsMonitor

    trainer = Trainer(callbacks=[DeviceStatsMonitor()])


Profiling
=========

Check out the :ref:`Profiler <profiler>` document.


----------------


****************
Model Statistics
****************

Print a Summary of Your LightningModule
=======================================

Whenever the ``.fit()`` function gets called, the Trainer will print the weights summary for the LightningModule.
By default it only prints the top-level modules. If you want to show all submodules in your network, use the
``max_depth`` option of :class:`~pytorch_lightning.callbacks.model_summary.ModelSummary` callback:

.. testcode::

    from pytorch_lightning.callbacks import ModelSummary

    trainer = Trainer(callbacks=[ModelSummary(max_depth=-1)])


You can also display the intermediate input- and output sizes of all your layers by setting the
``example_input_array`` attribute in your LightningModule. It will print a table like this

.. code-block:: text

      | Name  | Type        | Params | In sizes  | Out sizes
    --------------------------------------------------------------
    0 | net   | Sequential  | 132 K  | [10, 256] | [10, 512]
    1 | net.0 | Linear      | 131 K  | [10, 256] | [10, 512]
    2 | net.1 | BatchNorm1d | 1.0 K  | [10, 512] | [10, 512]

when you call ``.fit()`` on the Trainer. This can help you find bugs in the composition of your layers.

It is enabled by default and can be turned off using ``Trainer(enable_model_summary=False)``.

See Also:
    - :class:`~pytorch_lightning.callbacks.model_summary.ModelSummary`
    - :func:`~pytorch_lightning.utilities.model_summary.summarize`
    - :class:`~pytorch_lightning.utilities.model_summary.ModelSummary`


----------------


*************************************
Debugging with Distributed Strategies
*************************************

DDP Debugging
=============

If you are having a hard time debugging DDP on your remote machine you can debug DDP locally on the CPU. Note that this will not provide any speed benefits.

.. code-block:: python

    trainer = Trainer(accelerator="cpu", strategy="ddp", devices=2)

To inspect your code, you can use `pdb <https://docs.python.org/3/library/pdb.html>`_ or `breakpoint() <https://docs.python.org/3/library/functions.html#breakpoint>`_
or use regular print statements.

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):

            debugging_message = ...
            print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

            if self.trainer.global_rank == 0:
                import pdb

                pdb.set_trace()

            # to prevent other processes from moving forward until all processes are in sync
            self.trainer.strategy.barrier()

When everything works, switch back to GPU by changing only the accelerator.

.. code-block:: python

    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2)
