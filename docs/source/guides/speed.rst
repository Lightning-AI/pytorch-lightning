.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.core.lightning import LightningModule

.. _speed:

#######################
Speed up model training
#######################

<<<<<<< HEAD
There are multiple ways you can speed up your model's time to convergence:

* `<GPU/TPU training_>`_

* `<Mixed precision (16-bit) training_>`_

* `<Control Training Epochs_>`_

* `<Control Validation Frequency_>`_

* `<Limit Dataset Size_>`_

* `<Preload Data Into RAM_>`_

* `<Model Toggling_>`_

* `<Set Grads to None_>`_

* `<Things to avoid_>`_

****************
GPU/TPU training
****************

**Use when:** Whenever possible!

With Lightning, running on GPUs, TPUs or multiple node is a simple switch of a flag.

GPU training
============

Lightning supports a variety of plugins to further speed up distributed GPU training. Most notably:

* :class:`~pytorch_lightning.plugins.training_type.DDPPlugin`
* :class:`~pytorch_lightning.plugins.training_type.DDPShardedPlugin`
* :class:`~pytorch_lightning.plugins.training_type.DeepSpeedPlugin`

.. code-block:: python

    # run on 1 gpu
    trainer = Trainer(gpus=1)

    # train on 8 gpus, using DDP plugin
    trainer = Trainer(gpus=8, accelerator="ddp")

    # train on multiple GPUs across nodes (uses 8 gpus in total)
    trainer = Trainer(gpus=2, num_nodes=4)


GPU Training Speedup Tips
-------------------------

When training on single or multiple GPU machines, Lightning offers a host of advanced optimizations to improve throughput, memory efficiency, and model scaling.
Refer to :doc:`Advanced GPU Optimized Training for more details <../advanced/advanced_gpu>`.

Prefer DDP over DP
^^^^^^^^^^^^^^^^^^
:class:`~pytorch_lightning.plugins.training_type.DataParallelPlugin` performs three GPU transfers for EVERY batch:

1. Copy model to device.
2. Copy data to device.
3. Copy outputs of each device back to master.

Whereas :class:`~pytorch_lightning.plugins.training_type.DDPPlugin` only performs 1 transfer to sync gradients, making DDP MUCH faster than DP.


When using DDP plugins, set find_unused_parameters=False
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default we have set ``find_unused_parameters`` to True for compatibility reasons that have been observed in the past (see the `discussion <https://github.com/PyTorchLightning/pytorch-lightning/discussions/6219>`_ for more details).
This by default comes with a performance hit, and can be disabled in most cases.

.. tip::
    It applies to all DDP plugins that support ``find_unused_parameters`` as input.

.. code-block:: python

    from pytorch_lightning.plugins import DDPPlugin

    trainer = pl.Trainer(
        gpus=2,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

.. code-block:: python

    from pytorch_lightning.plugins import DDPSpawnPlugin

    trainer = pl.Trainer(
        gpus=2,
        plugins=DDPSpawnPlugin(find_unused_parameters=False),
    )

When using DDP on a multi-node cluster, set NCCL parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`NCCL <https://developer.nvidia.com/nccl>`__ is the NVIDIA Collective Communications Library which is used under the hood by PyTorch to handle communication across nodes and GPUs. There are reported benefits in terms of speedups when adjusting NCCL parameters as seen in this `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues/7179>`__. In the issue we see a 30% speed improvement when training the Transformer XLM-RoBERTa and a 15% improvement in training with Detectron2.

NCCL parameters can be adjusted via environment variables.

.. note::

    AWS and GCP already set default values for these on their clusters. This is typically useful for custom cluster setups.

* `NCCL_NSOCKS_PERTHREAD <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nsocks-perthread>`__
* `NCCL_SOCKET_NTHREADS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-nthreads>`__
* `NCCL_MIN_NCHANNELS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-min-nchannels>`__

.. code-block:: bash

    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_SOCKET_NTHREADS=2

Dataloaders
^^^^^^^^^^^
When building your DataLoader set ``num_workers > 0`` and ``pin_memory=True`` (only for GPUs).

.. code-block:: python

    Dataloader(dataset, num_workers=8, pin_memory=True)

num_workers
"""""""""""

The question of how many workers to specify in ``num_workers`` is tricky. Here's a summary of
some references, [`1 <https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813>`_], and our suggestions:

1. ``num_workers=0`` means ONLY the main process will load batches (that can be a bottleneck).
2. ``num_workers=1`` means ONLY one worker (just not the main process) will load data but it will still be slow.
3. The ``num_workers`` depends on the batch size and your machine.
4. A general place to start is to set ``num_workers`` equal to the number of CPU cores on that machine. You can get the number of CPU cores in python using `os.cpu_count()`, but note that depending on your batch size, you may overflow RAM memory.

.. warning:: Increasing ``num_workers`` will ALSO increase your CPU memory consumption.

The best thing to do is to increase the ``num_workers`` slowly and stop once you see no more improvement in your training speed.

Spawn
"""""
When using ``accelerator=ddp_spawn`` or training on TPUs, the way multiple GPUs/TPU cores are used is by calling ``.spawn()`` under the hood.
The problem is that PyTorch has issues with ``num_workers > 0`` when using ``.spawn()``. For this reason we recommend you
use ``accelerator=ddp`` so you can increase the ``num_workers``, however your script has to be callable like so:

.. code-block:: bash

    python my_program.py


TPU training
============

You can set the ``tpu_cores`` trainer flag to 1 or 8 cores.

.. code-block:: python

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

* You want to optimize for memory usage on a GPU.
* You have a GPU that supports 16 bit precision (NVIDIA pascal architecture or newer).
* Your optimization algorithm (training_step) is numerically stable.
* You want to be the cool person in the lab :p

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_precision.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+9+-+precision_1.mp4"></video>

|


Mixed precision combines the use of both 32 and 16 bit floating points to reduce memory footprint during model training, resulting in improved performance, achieving +3X speedups on modern GPUs.

Lightning offers mixed precision or 16-bit training for GPUs and TPUs.


.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    # 16-bit precision
    trainer = Trainer(precision=16, gpus=4)

=======
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
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde

----------------


***********************
<<<<<<< HEAD
Control Training Epochs
***********************

**Use when:** You run a hyperparameter search to find good initial parameters and want to save time, cost (money), or power (environment).
It can allow you to be more cost efficient and also run more experiments at the same time.

You can use Trainer flags to force training for a minimum number of epochs or limit to a max number of epochs. Use the `min_epochs` and `max_epochs` Trainer flags to set the number of epochs to run.
=======
Control Training epochs
***********************

It can be useful to force training for a minimum number of epochs or limit to a max number of epochs. Use the `min_epochs` and `max_epochs` Trainer flags to set the number of epochs to run.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde

.. testcode::

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)

<<<<<<< HEAD

If running iteration based training, i.e. infinite / iterable dataloader, you can also control the number of steps with the `min_steps` and  `max_steps` flags:

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
=======
----------------

****************************
Control validation frequency
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde
****************************

Check validation every n epochs
===============================
<<<<<<< HEAD

**Use when:** You have a small dataset, and want to run less validation checks.

You can limit validation check to only run every n epochs using the `check_val_every_n_epoch` Trainer flag.
=======
If you have a small dataset, you might want to check validation every n epochs. Use the `check_val_every_n_epoch` Trainer flag.
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde

.. testcode::

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)


Set validation check frequency within 1 training epoch
======================================================
<<<<<<< HEAD

**Use when:** You have a large training dataset, and want to run mid-epoch validation checks.

=======
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde
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

<<<<<<< HEAD
Learn more in our :ref:`trainer_flags` guide.

----------------

******************
Limit Dataset Size
=======
----------------

******************
Limit dataset size
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde
******************

Use data subset for training, validation, and test
==================================================
<<<<<<< HEAD

**Use when:** Debugging or running huge datasets.

If you don't want to check 100% of the training/validation/test set set these flags:
=======
If you don't want to check 100% of the training/validation/test set (for debugging or if it's huge), set these flags.
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde

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

<<<<<<< HEAD
Learn more in our :ref:`trainer_flags` guide.

-----

*********************
Preload Data Into RAM
*********************

**Use when:** You need access to all samples in a dataset at once.

When your training or preprocessing requires many operations to be performed on entire dataset(s), it can
sometimes be beneficial to store all data in RAM given there is enough space.
However, loading all data at the beginning of the training script has the disadvantage that it can take a long
time and hence it slows down the development process. Another downside is that in multiprocessing (e.g. DDP)
the data would get copied in each process.
One can overcome these problems by copying the data into RAM in advance.
Most UNIX-based operating systems provide direct access to tmpfs through a mount point typically named ``/dev/shm``.

0.  Increase shared memory if necessary. Refer to the documentation of your OS how to do this.

1.  Copy training data to shared memory:

    .. code-block:: bash

        cp -r /path/to/data/on/disk /dev/shm/

2.  Refer to the new data root in your script or command line arguments:

    .. code-block:: python

        datamodule = MyDataModule(data_root="/dev/shm/my_data")

---------

**************
Model Toggling
**************

**Use when:** Performing gradient accumulation with multiple optimizers in a
distributed setting.

Here is an explanation of what it does:

* Considering the current optimizer as A and all other optimizers as B.
* Toggling means that all parameters from B exclusive to A will have their ``requires_grad`` attribute set to ``False``.
* Their original state will be restored when exiting the context manager.

When performing gradient accumulation, there is no need to perform grad synchronization during the accumulation phase.
Setting ``sync_grad`` to ``False`` will block this synchronization and improve your training speed.

:class:`~pytorch_lightning.core.optimizer.LightningOptimizer` provides a
:meth:`~pytorch_lightning.core.optimizer.LightningOptimizer.toggle_model` function as a
:func:`contextlib.contextmanager` for advanced users.

Here is an example for advanced use-case:

.. testcode::

    # Scenario for a GAN with gradient accumulation every 2 batches and optimized for multiple gpus.
    class SimpleGAN(LightningModule):

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            # Implementation follows the PyTorch tutorial:
            # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            g_opt, d_opt = self.optimizers()

            X, _ = batch
            X.requires_grad = True
            batch_size = X.shape[0]

            real_label = torch.ones((batch_size, 1), device=self.device)
            fake_label = torch.zeros((batch_size, 1), device=self.device)

            # Sync and clear gradients
            # at the end of accumulation or
            # at the end of an epoch.
            is_last_batch_to_accumulate = \
                (batch_idx + 1) % 2 == 0 or self.trainer.is_last_batch

            g_X = self.sample_G(batch_size)

            ##########################
            # Optimize Discriminator #
            ##########################
            with d_opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
                d_x = self.D(X)
                errD_real = self.criterion(d_x, real_label)

                d_z = self.D(g_X.detach())
                errD_fake = self.criterion(d_z, fake_label)

                errD = (errD_real + errD_fake)

                self.manual_backward(errD)
                if is_last_batch_to_accumulate:
                    d_opt.step()
                    d_opt.zero_grad()

            ######################
            # Optimize Generator #
            ######################
            with g_opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
                d_z = self.D(g_X)
                errG = self.criterion(d_z, real_label)

                self.manual_backward(errG)
                if is_last_batch_to_accumulate:
                    g_opt.step()
                    g_opt.zero_grad()

            self.log_dict({'g_loss': errG, 'd_loss': errD}, prog_bar=True)

-----

*****************
Set Grads to None
*****************

In order to modestly improve performance, you can override :meth:`~pytorch_lightning.core.lightning.LightningModule.optimizer_zero_grad`.

For a more detailed explanation of pros / cons of this technique,
read `this <https://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.zero_grad>`_ documentation by the PyTorch team.

.. testcode::

    class Model(LightningModule):

        def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
            optimizer.zero_grad(set_to_none=True)


-----

***************
Things to avoid
***************

.item(), .numpy(), .cpu()
=========================
Don't call ``.item()`` anywhere in your code. Use ``.detach()`` instead to remove the connected graph calls. Lightning
takes a great deal of care to be optimized for this.

----------

empty_cache()
=============
Don't call this unnecessarily! Every time you call this ALL your GPUs have to wait to sync.

----------

Tranfering tensors to device
============================
LightningModules know what device they are on! Construct tensors on the device directly to avoid CPU->Device transfer.

.. code-block:: python

    # bad
    t = torch.rand(2, 2).cuda()

    # good (self is LightningModule)
    t = torch.rand(2, 2, device=self.device)


For tensors that need to be model attributes, it is best practice to register them as buffers in the modules's
``__init__`` method:

.. code-block:: python

    # bad
    self.t = torch.rand(2, 2, device=self.device)

    # good
    self.register_buffer("t", torch.rand(2, 2))
=======

--------------

******************************
Transfer Learning (finetuning)
******************************


Finetuning callbacks
====================

TODO: add some missing context

.. autoclass:: pytorch_lightning.callbacks.BaseFinetuning
    :members:
    :noindex:


.. autoclass:: pytorch_lightning.callbacks.BackboneFinetuning
    :members:
    :noindex:

Using Pretrained Models
=======================

You can use any :class:`~pytorch_lightning.core.lightning.LightningModule` as a pretrained model, since any LightningModule is EXACTLY a `torch.nn.Module` but with more capabilities.

Example: Autoencoder
--------------------

For example, we can use a pretrained Lightning AutoEncoder as a feature extractor for another model:


.. testcode::

    class Encoder(torch.nn.Module):
        ...

    class AutoEncoder(LightningModule):
        def __init__(self):
            self.encoder = Encoder()
            self.decoder = Decoder()

    class CIFAR10Classifier(LightningModule):
        def __init__(self):
            # init the pretrained LightningModule
            self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH)
            self.feature_extractor.freeze()

            # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
            self.classifier = nn.Linear(100, 10)

        def forward(self, x):
            representations = self.feature_extractor(x)
            x = self.classifier(representations)
            ...


Example: Imagenet (computer Vision)
-----------------------------------

Using a pretrained model on imagenet, finetuned on CIFAR-10 to predict on CIFAR-10.

.. testcode::
    :skipif: not _TORCHVISION_AVAILABLE

    import torchvision.models as models

    class ImagenetTransferLearning(LightningModule):
        def __init__(self):
            super().__init__()

            # init a pretrained resnet
            backbone = models.resnet50(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)

            # use the pretrained model to classify cifar-10 (10 image classes)
            num_target_classes = 10
            self.classifier = nn.Linear(num_filters, num_target_classes)

        def forward(self, x):
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
            ...

.. code-block:: python

    # finetune
    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)


    # predict on your data of interest
    model = ImagenetTransferLearning.load_from_checkpoint(PATH)
    model.freeze()

    x = some_images_from_cifar10()
    predictions = model(x)

Example: BERT (NLP)
-------------------
Lightning is completely agnostic to what's used for transfer learning so long
as it is a `torch.nn.Module` subclass.

Here's a model that uses `Huggingface transformers <https://github.com/huggingface/transformers>`_.

.. testcode::

    class BertMNLIFinetuner(LightningModule):

        def __init__(self):
            super().__init__()

            self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
            self.W = nn.Linear(bert.config.hidden_size, 3)
            self.num_classes = 3


        def forward(self, input_ids, attention_mask, token_type_ids):

            h, _, attn = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

            h_cls = h[:, 0]
            logits = self.W(h_cls)
            return logits, attn
>>>>>>> d5728ab49409cd9df2a7ca22e60055d9940d2dde
