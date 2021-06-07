.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.core.lightning import LightningModule

.. _speed:

#######################
Speed up model training
#######################

There are multiple ways you can speed up your model's time to convergence:

* `<Early stopping_>`_

* `<GPU/TPU training_>`_

* `<Mixed precision (16-bit) training_>`_

* `<Control Training Epochs_>`_

* `<Control Validation Frequency_>`_

* `<Limit Dataset Size_>`_

.. _early_stopping:

**************
Early stopping
**************

**Use when:**

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_earlystop.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+19-+early+stopping_1.mp4"></video>

|

Early stopping is an optimization technique used to avoid overfitting without compromising the model's accuracy. When training a large network, there will be a point during training when the model will stop generalizing and start learning the statistical noise in the training dataset. To avoid overfitting and reduce training time, unable early stopping to stop training at the point when performance on a validation dataset starts to degrade (you can pick the metric to monitor).


Stop training when metric is degrading
======================================
The
:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
callback can be used to monitor a validation metric and stop the training when no improvement is observed.

.. testcode::

    # 1. Import EarlyStopping callback
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    # 2. Log the metric you want to monitor using `self.log` method.
    def validation_step(...):
        self.log('val_loss', loss)

    # 3. Init the callback, and set `monitor` to the logged metric of your choice.
    early_stop_callback = EarlyStopping(monitor='val_loss');

    # 4. Pass the callback to the Trainer `callbacks` flag.
    trainer = Trainer(callbacks=[early_stop_callback])


Stop training based on metric value
===================================
You can set a `stopping_threshold` to stop training immediately once the monitored quantity reaches this threshold. It is useful when we know that going beyond a certain optimal value does not further benefit us.

.. testcode::

    early_stop_callback = EarlyStopping(
       monitor='val_accuracy',
       stopping_threshold=0.98

    )
    trainer = Trainer(callbacks=[early_stop_callback])

You can set a `divergence_threshold` to stop training as soon as the monitored quantity is lower than this threshold. When reaching a value this bad, we believe the model cannot recover anymore and it is better to stop early and run with different initial conditions.

You can also set `check_finite` to stop training when the monitored metric becomes NaN or infinite.

Learn more in the :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` doc.


Stopping an epoch early
=======================

You can stop an epoch early by overriding :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_train_batch_start` to return ``-1`` when some condition is met.

If you do this repeatedly, for every epoch you had originally requested, then this will stop your entire run.

----------

****************
GPU/TPU training
****************

**Use when:** Running large datasets or want to speed up your training.

With Lightning, running on GPUs, TPUs or multiple node is a simple switch of a flag.

GPU training
============

Lightning supports a variety of plugins to further speed up distributed GPU training. Most notably:

* :class:`~pytorch_lightning.plugins.training_type.DDPPlugin`
* :class:`~pytorch_lightning.plugins.training_type.DDPShardedPlugin`
* :class:`~pytorch_lightning.plugins.training_type.DeepSpeedPlugin`

.. testcode::

    # run on 1 gpu
    trainer = Trainer(gpus=1)

    # train on 8 gpus
    trainer = Trainer(gpus=8)

    # train on multiple GPUs across nodes (uses 8 gpus in total)
    trainer = Trainer(gpus=2, num_nodes=4)


TPU training
============

.. testcode::

    # train on 1 TPU core
    trainer = Trainer(tpu_cores=1)

    # train on 8 TPU cores
    trainer = Trainer(tpu_cores=8)

To train on more than 8 cores (ie: a POD),
submit this script using the xla_dist script.

Example::

    python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    --env=XLA_USE_BF16=1
    -- python your_trainer_file.py


Read more in our :ref:`accelerators` and :ref:`plugins` guides.


-----------

.. _amp:

*********************************
Mixed precision (16-bit) training
*********************************

**Use when:**

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_precision.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+9+-+precision_1.mp4"></video>

|


Mixed precision is the combined use of both 32 and 16 bit floating points to reduce memory footprint during model training, resulting in improved performance, achieving +3X speedups on modern GPUs.

Lightning offers mixed precision or 16-bit training for CPUs, GPUs, and TPUs. 


.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    # 16-bit precision
    trainer = Trainer(precision=16, gpus=4)


----------------


***********************
Control Training Epochs
***********************

**Use when:**

It can be useful to force training for a minimum number of epochs or limit to a max number of epochs. Use the `min_epochs` and `max_epochs` Trainer flags to set the number of epochs to run.

.. testcode::

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)


You can also control the number of steos with the `min_steps` and  `max_steps` flags:

.. testcode::

    trainer = Trainer(max_steps=1000)

    trainer = Trainer(min_steps=100)

You can also interupt training based on training time:

.. testcode::
    
    # Stop after 12 hours of training or when reaching 10 epochs (string)
    trainer = Trainer(max_time="00:12:00:00", max_epochs=10)

    # Stop after 1 day and 5 hours (dict)
    trainer = Trainer(max_time={"days": 1, "hours": 5})

Learn more in our :ref:`trainer_flags` guide.


----------------

****************************
Control Validation Frequency
****************************

Check validation every n epochs
===============================

**Use when:** You have a small dataset, and want to run less validation checks.

You can limit validation check to only run every n epochs using the `check_val_every_n_epoch` Trainer flag.

.. testcode::

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)


Set validation check frequency within 1 training epoch
======================================================

**Use when:** You have a large dataset, and want to run mid-epoch validation checks.

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

Learn more in our :ref:`trainer_flags` guide.

----------------

******************
Limit Dataset Size
******************

Use data subset for training, validation, and test
==================================================

**Use when:** Debugging or running huge datasets.

If you don't want to check 100% of the training/validation/test set set these flags:

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

Learn more in our :ref:`trainer_flags` guide.


