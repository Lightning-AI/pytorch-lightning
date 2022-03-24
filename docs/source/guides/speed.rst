.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.core.lightning import LightningModule

.. _training-speedup:


#######################
Speed Up Model Training
#######################

When you are limited with the resources, it becomes hard to speed up model training and reduce the training time
without affecting the model's performance. There are multiple ways you can speed up your model's time to convergence.


************************
Training on Accelerators
************************

**Use when:** Whenever possible!

With Lightning, running on GPUs, TPUs, IPUs on multiple nodes is a simple switch of a flag.

GPU Training
============

Lightning supports a variety of plugins to speed up distributed GPU training. Most notably:

* :class:`~pytorch_lightning.strategies.DDPStrategy`
* :class:`~pytorch_lightning.strategies.DDPShardedStrategy`
* :class:`~pytorch_lightning.strategies.DeepSpeedStrategy`

.. code-block:: python

    # run on 1 gpu
    trainer = Trainer(accelerator="gpu", devices=1)

    # train on 8 gpus, using the DDP strategy
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp")

    # train on multiple GPUs across nodes (uses 8 gpus in total)
    trainer = Trainer(accelerator="gpu", devices=2, num_nodes=4)


GPU Training Speedup Tips
-------------------------

When training on single or multiple GPU machines, Lightning offers a host of advanced optimizations to improve throughput, memory efficiency, and model scaling.
Refer to :doc:`Advanced GPU Optimized Training for more details <../advanced/model_parallel>`.

Prefer DDP Over DP
^^^^^^^^^^^^^^^^^^
:class:`~pytorch_lightning.strategies.dp.DataParallelStrategy` performs three GPU transfers for EVERY batch:

1. Copy the model to the device.
2. Copy the data to the device.
3. Copy the outputs of each device back to the main device.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/distributed_training/dp.gif
    :alt: Animation showing DP execution.
    :width: 500
    :align: center

|

Whereas :class:`~pytorch_lightning.strategies.ddp.DDPStrategy` only performs two transfer operations, making DDP much faster than DP:

1. Moving data to the device.
2. Transfer and sync gradients.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/distributed_training/ddp.gif
    :alt: Animation showing DDP execution.
    :width: 500
    :align: center

|


When Using DDP Plugins, Set find_unused_parameters=False
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, we have set ``find_unused_parameters=True`` for compatibility reasons that have been observed in the past (refer to the `discussion <https://github.com/PyTorchLightning/pytorch-lightning/discussions/6219>`_ for more details).
When enabled, it can result in a performance hit and can be disabled in most cases. Read more about it `here <https://pytorch.org/docs/stable/notes/ddp.html#internal-design>`_.

.. tip::
    It applies to all DDP strategies that support ``find_unused_parameters`` as input.

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        gpus=2,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

.. code-block:: python

    from pytorch_lightning.strategies import DDPSpawnStrategy

    trainer = pl.Trainer(
        gpus=2,
        strategy=DDPSpawnStrategy(find_unused_parameters=False),
    )

When Using DDP on a Multi-node Cluster, Set NCCL Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`NCCL <https://developer.nvidia.com/nccl>`__ is the NVIDIA Collective Communications Library that is used by PyTorch to handle communication across nodes and GPUs. There are reported benefits in terms of speedups when adjusting NCCL parameters as seen in this `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues/7179>`__. In the issue, we see a 30% speed improvement when training the Transformer XLM-RoBERTa and a 15% improvement in training with Detectron2.

NCCL parameters can be adjusted via environment variables.

.. note::

    AWS and GCP already set default values for these on their clusters. This is typically useful for custom cluster setups.

* `NCCL_NSOCKS_PERTHREAD <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nsocks-perthread>`__
* `NCCL_SOCKET_NTHREADS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-nthreads>`__
* `NCCL_MIN_NCHANNELS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-min-nchannels>`__

.. code-block:: bash

    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_SOCKET_NTHREADS=2

DataLoaders
^^^^^^^^^^^

When building your DataLoader set ``num_workers>0`` and ``pin_memory=True`` (only for GPUs).

.. code-block:: python

    Dataloader(dataset, num_workers=8, pin_memory=True)

num_workers
^^^^^^^^^^^

The question of how many workers to specify in ``num_workers`` is tricky. Here's a summary of `some references <https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813>`_, and our suggestions:

1. ``num_workers=0`` means ONLY the main process will load batches (that can be a bottleneck).
2. ``num_workers=1`` means ONLY one worker (just not the main process) will load data, but it will still be slow.
3. The performance of high ``num_workers`` depends on the batch size and your machine.
4. A general place to start is to set ``num_workers`` equal to the number of CPU cores on that machine. You can get the number of CPU cores in python using ``os.cpu_count()``, but note that depending on your batch size, you may overflow RAM memory.

.. warning:: Increasing ``num_workers`` will ALSO increase your CPU memory consumption.

The best thing to do is to increase the ``num_workers`` slowly and stop once there is no more improvement in your training speed.

For debugging purposes or for dataloaders that load very small datasets, it is desirable to set ``num_workers=0``. However, this will always log a warning for every dataloader with ``num_workers <= min(2, os.cpu_count())``. In such cases, you can specifically filter this warning by using:

.. code-block:: python

    import warnings

    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    # or to ignore all warnings that could be false positives
    from pytorch_lightning.utilities.warnings import PossibleUserWarning

    warnings.filterwarnings("ignore", category=PossibleUserWarning)

Spawn
^^^^^

When using ``strategy="ddp_spawn"`` or training on TPUs, the way multiple GPUs/TPU cores are used is by calling :obj:`torch.multiprocessing`
``.spawn()`` under the hood. The problem is that PyTorch has issues with ``num_workers>0`` when using ``.spawn()``. For this reason, we recommend you
use ``strategy="ddp"`` so you can increase the ``num_workers``, however since DDP doesn't work in an interactive environment like IPython/Jupyter notebooks
your script has to be callable like so:

.. code-block:: bash

    python my_program.py

However, using ``strategy="ddp_spawn"`` enables to reduce memory usage with In-Memory Dataset and shared memory tensors. For more info, checkout
:ref:`Sharing Datasets Across Process Boundaries <ddp_spawn_shared_memory>` section.

Persistent Workers
^^^^^^^^^^^^^^^^^^

When using ``strategy="ddp_spawn"`` and ``num_workers>0``, consider setting ``persistent_workers=True`` inside your DataLoader since it can result in data-loading bottlenecks and slowdowns.
This is a limitation of Python ``.spawn()`` and PyTorch.


TPU Training
============

You can set the ``tpu_cores`` trainer flag to 1, [7] (specific core) or eight cores.

.. code-block:: python

    # train on 1 TPU core
    trainer = Trainer(tpu_cores=1)

    # train on 7th TPU core
    trainer = Trainer(tpu_cores=[7])

    # train on 8 TPU cores
    trainer = Trainer(tpu_cores=8)

To train on more than eight cores (a POD),
submit this script using the xla_dist script.

Example::

    python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    --env=XLA_USE_BF16=1
    -- python your_trainer_file.py


Read more in our :ref:`accelerators` and :ref:`plugins` guides.


-----------

**************
Early Stopping
**************

Usually, long training epochs can lead to either overfitting or no major improvements in your metrics due to no limited convergence.
Here :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback can help you stop the training entirely by monitoring a metric of your choice.

You can read more about it :ref:`here <early_stopping>`.

----------

.. _speed_amp:

*********************************
Mixed Precision (16-bit) Training
*********************************

Lower precision, such as the 16-bit floating-point, enables the training and deployment of large neural networks since they require less memory, enhance data transfer operations since they required
less memory bandwidth and run match operations much faster on GPUs that support Tensor Core.

**Use when:**

* You want to optimize for memory usage on a GPU.
* You have a GPU that supports 16-bit precision (NVIDIA pascal architecture or newer).
* Your optimization algorithm (training_step) is numerically stable.
* You want to be the cool person in the lab :p

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_precision.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+9+-+precision_1.mp4"></video>

|

Mixed precision combines the use of both 32 and 16-bit floating points to reduce memory footprint during model training, resulting in improved performance, achieving upto +3X speedups on modern GPUs.

Lightning offers mixed precision training for GPUs and CPUs, as well as bfloat16 mixed precision training for TPUs.


.. testcode::
    :skipif: torch.cuda.device_count() < 4

    # 16-bit precision
    trainer = Trainer(precision=16, gpus=4)


Read more about :ref:`mixed-precision training <amp>`.


----------------


***********************
Control Training Epochs
***********************

**Use when:** You run a hyperparameter search to find good initial parameters and want to save time, cost (money), or power (environment).
It can allow you to be more cost efficient and also run more experiments at the same time.

You can use Trainer flags to force training for a minimum number of epochs or limit it to a max number of epochs. Use the ``min_epochs`` and ``max_epochs`` Trainer flags to set the number of epochs to run.
Setting ``min_epochs=N`` makes sure that the training will run for at least ``N`` epochs. Setting ``max_epochs=N`` will ensure that training won't happen after
``N`` epochs.

.. testcode::

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)


If running iteration based training, i.e., infinite / iterable DataLoader, you can also control the number of steps with the ``min_steps`` and  ``max_steps`` flags:

.. testcode::

    trainer = Trainer(max_steps=1000)

    trainer = Trainer(min_steps=100)

You can also interrupt training based on training time:

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

Check Validation Every n Epochs
===============================

**Use when:** You have a small dataset and want to run fewer validation checks.

You can limit validation check to only run every n epochs using the ``check_val_every_n_epoch`` Trainer flag.

.. testcode::

    # default
    trainer = Trainer(check_val_every_n_epoch=1)

    # runs validation after every 7th Epoch
    trainer = Trainer(check_val_every_n_epoch=7)


Validation Within Training Epoch
================================

**Use when:** You have a large training dataset and want to run mid-epoch validation checks.

For large datasets, it's often desirable to check validation multiple times within a training epoch.
Pass in a float to check that often within one training epoch. Pass in an int ``K`` to check every ``K`` training batch.
Must use an ``int`` if using an :class:`~torch.utils.data.IterableDataset`.

.. testcode::

    # default
    trainer = Trainer(val_check_interval=1.0)

    # check every 1/4 th of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for IterableDatasets or fixed frequency)
    trainer = Trainer(val_check_interval=100)

Learn more in our :ref:`trainer_flags` guide.

----------------

*********************
Preload Data Into RAM
*********************

**Use when:** You need access to all samples in a dataset at once.

When your training or preprocessing requires many operations to be performed on entire dataset(s), it can
sometimes be beneficial to store all data in RAM given there is enough space.
However, loading all data at the beginning of the training script has the disadvantage that it can take a long
time, and hence, it slows down the development process. Another downside is that in multiprocessing (e.g., DDP)
the data would get copied in each process.
One can overcome these problems by copying the data into RAM in advance.
Most UNIX-based operating systems provide direct access to tmpfs through a mount point typically named ``/dev/shm``.

Increase shared memory if necessary. Refer to the documentation of your OS on how to do this.

1.  Copy training data to shared memory:

    .. code-block:: bash

        cp -r /path/to/data/on/disk /dev/shm/

2.  Refer to the new data root in your script or command-line arguments:

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
* Toggling, which means all parameters from B exclusive to A will have their ``requires_grad`` attribute set to ``False``.
* Restoring their original state when exiting the context manager.

When performing gradient accumulation, there is no need to perform grad synchronization during the accumulation phase.
Setting ``sync_grad`` to ``False`` will block this synchronization and improve your training speed.

:class:`~pytorch_lightning.core.optimizer.LightningOptimizer` provides a
:meth:`~pytorch_lightning.core.optimizer.LightningOptimizer.toggle_model` function as a
:func:`contextlib.contextmanager` for advanced users.

Here is an example of an advanced use case:

.. testcode::

    # Scenario for a GAN with gradient accumulation every two batches and optimized for multiple gpus.
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
            is_last_batch_to_accumulate = (batch_idx + 1) % 2 == 0 or self.trainer.is_last_batch

            g_X = self.sample_G(batch_size)

            ##########################
            # Optimize Discriminator #
            ##########################
            with d_opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
                d_x = self.D(X)
                errD_real = self.criterion(d_x, real_label)

                d_z = self.D(g_X.detach())
                errD_fake = self.criterion(d_z, fake_label)

                errD = errD_real + errD_fake

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

            self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

-----

*****************
Set Grads to None
*****************

In order to improve performance, you can override :meth:`~pytorch_lightning.core.lightning.LightningModule.optimizer_zero_grad`.

For a more detailed explanation of the pros / cons of this technique,
read the documentation for :meth:`~torch.optim.Optimizer.zero_grad` by the PyTorch team.

.. testcode::

    class Model(LightningModule):
        def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
            optimizer.zero_grad(set_to_none=True)


-----

***************
Things to Avoid
***************

.item(), .numpy(), .cpu()
=========================

Don't call ``.item()`` anywhere in your code. Use ``.detach()`` instead to remove the connected graph calls. Lightning
takes a great deal of care to be optimized for this.

Clear Cache
===========

Don't call :func:`torch.cuda.empty_cache` unnecessarily! Every time you call this, ALL your GPUs have to wait to sync.

Transferring Tensors to Device
==============================

LightningModules know what device they are on! Construct tensors on the device directly to avoid CPU->Device transfer.

.. code-block:: python

    # bad
    t = torch.rand(2, 2).cuda()

    # good (self is LightningModule)
    t = torch.rand(2, 2, device=self.device)


For tensors that need to be model attributes, it is best practice to register them as buffers in the module's
``__init__`` method:

.. code-block:: python

    # bad
    self.t = torch.rand(2, 2, device=self.device)

    # good
    self.register_buffer("t", torch.rand(2, 2))
