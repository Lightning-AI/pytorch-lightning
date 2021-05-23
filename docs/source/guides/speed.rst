.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.core.lightning import LightningModule

.. _speed:

#######################
Speed up model training
#######################

There are multiple ways you can speed up your model.

.. _early_stopping:

**************
Early stopping
**************

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_earlystop.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+19-+early+stopping_1.mp4"></video>

|

Early stopping based on metric
==============================
The
:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
callback can be used to monitor a validation metric and stop the training when no improvement is observed.

To enable it:

- Import :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback.
- Log the metric you want to monitor using :func:`~pytorch_lightning.core.lightning.LightningModule.log` method.
- Init the callback, and set `monitor` to the logged metric of your choice.
- Pass the :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback to the :class:`~pytorch_lightning.trainer.trainer.Trainer` callbacks flag.

.. code-block:: python

    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    def validation_step(...):
        self.log('val_loss', loss)

    trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss')])

You can customize the callbacks behaviour by changing its parameters.

.. testcode::

    early_stop_callback = EarlyStopping(
       monitor='val_accuracy',
       min_delta=0.00,
       patience=3,
       verbose=False,
       mode='max'
    )
    trainer = Trainer(callbacks=[early_stop_callback])


Additional parameters that stop training at extreme points:

- ``stopping_threshold``: Stops training immediately once the monitored quantity reaches this threshold.
  It is useful when we know that going beyond a certain optimal value does not further benefit us.
- ``divergence_threshold``: Stops training as soon as the monitored quantity becomes worse than this threshold.
  When reaching a value this bad, we believe the model cannot recover anymore and it is better to stop early and run with different initial conditions.
- ``check_finite``: When turned on, we stop training if the monitored metric becomes NaN or infinite.

In case you need early stopping in a different part of training, subclass :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
and change where it is called:

.. testcode::

    class MyEarlyStopping(EarlyStopping):

        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer, pl_module)

.. note::
   The :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback runs
   at the end of every validation epoch,
   which, under the default configuration, happen after every training epoch.
   However, the frequency of validation can be modified by setting various parameters
   in the :class:`~pytorch_lightning.trainer.trainer.Trainer`,
   for example :paramref:`~pytorch_lightning.trainer.trainer.Trainer.check_val_every_n_epoch`
   and :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval`.
   It must be noted that the `patience` parameter counts the number of
   validation epochs with no improvement, and not the number of training epochs.
   Therefore, with parameters `check_val_every_n_epoch=10` and `patience=3`, the trainer
   will perform at least 40 training epochs before being stopped.

.. seealso::
    - :class:`~pytorch_lightning.trainer.trainer.Trainer`
    - :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`


Stopping an epoch early
=======================

You can stop an epoch early by overriding :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_train_batch_start` to return ``-1`` when some condition is met.

If you do this repeatedly, for every epoch you had originally requested, then this will stop your entire run.

----------

.. _amp:

*********************************
Mixed precision (16-bit) training
*********************************

Mixed precision is the combined use of both 32 and 16 bit floating points during model training, which reduced memory requirements and improves performance significantly, achiving over 3X speedups on modern GPUs.

Lightning offers mixed precision or 16-bit training for CPUs, GPUs, and TPUs.

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_precision.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+9+-+precision_1.mp4"></video>

|


16-bit precision on GPUs
========================
Mixed or 16-bit precision can cut your memory footprint by half.
If using volta architecture GPUs it can give a dramatic training speed-up as well.

When using PyTorch 1.6+, Lightning uses the native AMP implementation to support 16-bit precision.

.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    # turn on 16-bit precision
    trainer = Trainer(precision=16, gpus=1)

.. admonition:: Using 16-bit precision with PyTorch < 1.6 is not recommended, but supported using apex.
   :class: dropdown, warning

    NVIDIA Apex and DDP have instability problems. We recommend upgrading to PyTorch 1.6+ to use the native AMP 16-bit precision.

    If you are using an earlier version of PyTorch (before 1.6), Lightning uses `Apex <https://github.com/NVIDIA/apex>`_ to support 16-bit training.

    To use Apex 16-bit training:

    1. Install Apex

    .. code-block:: bash

        # ------------------------
        # OPTIONAL: on your cluster you might need to load CUDA 10 or 9
        # depending on how you installed PyTorch

        # see available modules
        module avail

        # load correct CUDA before install
        module load cuda-10.0
        # ------------------------

        # make sure you've loaded a cuda version > 4.0 and < 7.0
        module load gcc-6.1.0

        $ pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex

    2. Set the `precision` trainer flag to 16. You can customize the `Apex optimization level <https://nvidia.github.io/apex/amp.html#opt-levels>`_ by setting the `amp_level` flag.

    .. testcode::
        :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

        # turn on 16-bit
        trainer = Trainer(amp_backend="apex", amp_level='O2', precision=16)

    If you need to configure the apex init for your particular use case, or want to ucustumize the
    16-bit training behviour, override :meth:`pytorch_lightning.core.LightningModule.configure_apex`.

16-bit precision on TPUs
========================
To use 16-bit precision on TPUs simply set the number of tpu cores, and set `precision` trainer flag to 16.

.. testcode::
    :skipif: not _TPU_AVAILABLE

    # DEFAULT
    trainer = Trainer(tpu_cores=8, precision=32)

    # turn on 16-bit
    trainer = Trainer(tpu_cores=8, precision=16)

----------------


***********************
Control Training epochs
***********************

It can be useful to force training for a minimum number of epochs or limit to a max number of epochs. Use the `min_epochs` and `max_epochs` Trainer flags to set the number of epochs to run.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)

----------------

****************************
Control validation frequency
****************************

Check validation every n epochs
===============================
If you have a small dataset, you might want to check validation every n epochs. Use the `check_val_every_n_epoch` Trainer flag.

.. testcode::

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)


Set validation check frequency within 1 training epoch
======================================================
For large datasets, it's often desirable to check validation multiple times within a training loop.
Pass in a float to check that often within 1 training epoch. Pass in an int `k` to check every `k` training batches.
Must use an `int` if using an `IterableDataset`.

.. testcode::

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for `IterableDatasets` or fixed frequency)
    trainer = Trainer(val_check_interval=100)

----------------

******************
Limit dataset size
******************

Use data subset for training, validation, and test
==================================================
If you don't want to check 100% of the training/validation/test set (for debugging or if it's huge), set these flags.

.. testcode::

    # DEFAULT
    trainer = Trainer(
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0
    )

    # check 10%, 20%, 30% only, respectively for training, validation and test set
    trainer = Trainer(
        limit_train_batches=0.1,
        limit_val_batches=0.2,
        limit_test_batches=0.3
    )

If you also pass ``shuffle=True`` to the dataloader, a different random subset of your dataset will be used for each epoch; otherwise the same subset will be used for all epochs.

.. note:: ``limit_train_batches``, ``limit_val_batches`` and ``limit_test_batches`` will be overwritten by ``overfit_batches`` if ``overfit_batches`` > 0. ``limit_val_batches`` will be ignored if ``fast_dev_run=True``.

.. note:: If you set ``limit_val_batches=0``, validation will be disabled.
