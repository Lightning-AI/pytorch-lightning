###################
What is a Strategy?
###################

Strategy controls the model distribution across training, evaluation, and prediction to be used by the :doc:`Trainer <../common/trainer>`. It can be controlled by passing different
strategy with aliases (``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"`` and so on) as well as a custom strategy to the ``strategy`` parameter for Trainer.

The Strategy in PyTorch Lightning handles the following responsibilities:

* Launch and teardown of training processes (if applicable).
* Setup communication between processes (NCCL, GLOO, MPI, and so on).
* Provide a unified communication interface for reduction, broadcast, and so on.
* Owns the :class:`~pytorch_lightning.core.module.LightningModule`
* Handles/owns optimizers and schedulers.


Strategy is a composition of one :doc:`Accelerator <../extensions/accelerator>`, one :ref:`Precision Plugin <extensions/plugins:Precision Plugins>`, a :ref:`CheckpointIO <extensions/plugins:CheckpointIO Plugins>`
plugin and other optional plugins such as the :ref:`ClusterEnvironment <extensions/plugins:Cluster Environments>`.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/overview.jpeg
    :alt: Illustration of the Strategy as a composition of the Accelerator and several plugins

We expose Strategies mainly for expert users that want to extend Lightning for new hardware support or new distributed backends (e.g. a backend not yet supported by `PyTorch <https://pytorch.org/docs/stable/distributed.html#backends>`_ itself).


----------

*****************************
Selecting a Built-in Strategy
*****************************

Built-in strategies can be selected in two ways.

1. Pass the shorthand name to the ``strategy`` Trainer argument
2. Import a Strategy from :mod:`pytorch_lightning.strategies`, instantiate it and pass it to the ``strategy`` Trainer argument

The latter allows you to configure further options on the specifc strategy.
Here are some examples:


.. code-block:: python

    # Training with the DistributedDataParallel strategy on 4 GPUs
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with the DistributedDataParallel strategy on 4 GPUs, with options configured
    trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=4)

    # Training with the DDP Spawn strategy using auto accelerator selection
    trainer = Trainer(strategy="ddp_spawn", accelerator="auto", devices=4)

    # Training with the DeepSpeed strategy on available GPUs
    trainer = Trainer(strategy="deepspeed", accelerator="gpu", devices="auto")

    # Training with the DDP strategy using 3 CPU processes
    trainer = Trainer(strategy="ddp", accelerator="cpu", devices=3)

    # Training with the DDP Spawn strategy on 8 TPU cores
    trainer = Trainer(strategy="ddp_spawn", accelerator="tpu", devices=8)

    # Training with the default IPU strategy on 8 IPUs
    trainer = Trainer(accelerator="ipu", devices=8)


The below table lists all relevant strategies available in Lightning with their corresponding short-hand name:


.. list-table:: Strategy Classes and Nicknames
   :widths: 20 20 20
   :header-rows: 1

   * - Name
     - Class
     - Description
   * - bagua
     - :class:`~pytorch_lightning.strategies.BaguaStrategy`
     - Strategy for training using the Bagua library, with advanced distributed training algorithms and system optimizations. :ref:`Learn more. <accelerators/gpu_intermediate:Bagua>`
   * - collaborative
     - :class:`~pytorch_lightning.strategies.CollaborativeStrategy`
     - Strategy for training collaboratively on local machines or unreliable GPUs across the internet. :ref:`Learn more. <strategies/collaborative_training:Training on unreliable mixed GPUs across the internet>`
   * - fsdp
     - :class:`~pytorch_lightning.strategies.DDPFullyShardedStrategy`
     - Strategy for Fully Sharded Data Parallel provided by FairScale. :ref:`Learn more. <advanced/model_parallel:Fully Sharded Training>`
   * - ddp_sharded
     - :class:`~pytorch_lightning.strategies.DDPShardedStrategy`
     - Optimizer and gradient sharded training provided by FairScale. :ref:`Learn more. <advanced/model_parallel:Sharded Training>`
   * - ddp_sharded_spawn
     - :class:`~pytorch_lightning.strategies.DDPSpawnShardedStrategy`
     - Optimizer sharded training provided by FairScale. :ref:`Learn more. <advanced/model_parallel:Sharded Training>`
   * - ddp_spawn
     - :class:`~pytorch_lightning.strategies.DDPSpawnStrategy`
     - Spawns processes using the :func:`torch.multiprocessing.spawn` method and joins processes after training finishes. :ref:`Learn more. <accelerators/gpu_intermediate:Distributed Data Parallel Spawn>`
   * - ddp
     - :class:`~pytorch_lightning.strategies.DDPStrategy`
     - Strategy for multi-process single-device training on one or multiple nodes. :ref:`Learn more. <accelerators/gpu_intermediate:Distributed Data Parallel>`
   * - dp
     - :class:`~pytorch_lightning.strategies.DataParallelStrategy`
     - Implements data-parallel training in a single process, i.e., the model gets replicated to each device and each gets a split of the data. :ref:`Learn more. <accelerators/gpu_intermediate:Data Parallel>`
   * - deepspeed
     - :class:`~pytorch_lightning.strategies.DeepSpeedStrategy`
     - Provides capabilities to run training using the DeepSpeed library, with training optimizations for large billion parameter models. :ref:`Learn more. <advanced/model_parallel:deepspeed>`
   * - horovod
     - :class:`~pytorch_lightning.strategies.HorovodStrategy`
     - Strategy for Horovod distributed training integration. :ref:`Learn more. <accelerators/gpu_intermediate:Horovod>`
   * - hpu_parallel
     - :class:`~pytorch_lightning.strategies.HPUParallelStrategy`
     - Strategy for distributed training on multiple HPU devices. :doc:`Learn more. <../accelerators/hpu>`
   * - hpu_single
     - :class:`~pytorch_lightning.strategies.SingleHPUStrategy`
     - Strategy for training on a single HPU device. :doc:`Learn more. <../accelerators/hpu>`
   * - ipu_strategy
     - :class:`~pytorch_lightning.strategies.IPUStrategy`
     - Plugin for training on IPU devices. :doc:`Learn more. <../accelerators/ipu>`
   * - tpu_spawn
     - :class:`~pytorch_lightning.strategies.TPUSpawnStrategy`
     - Strategy for training on multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn` method. :doc:`Learn more. <../accelerators/tpu>`
   * - single_tpu
     - :class:`~pytorch_lightning.strategies.SingleTPUStrategy`
     - Strategy for training on a single TPU device. :doc:`Learn more. <../accelerators/tpu>`


************************
Create a Custom Strategy
************************

Every strategy in Lightning is a subclass of one of the main base classes: :class:`~pytorch_lightning.strategies.Strategy`, :class:`~pytorch_lightning.strategies.SingleDeviceStrategy` or :class:`~pytorch_lightning.strategies.ParallelStrategy`.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/hierarchy.jpeg
    :alt: Strategy base classes

As an expert user, you may choose to extend either an existing built-in Strategy or create a completely new one by
subclassing the base classes.

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy


    class CustomDDPStrategy(DDPStrategy):
        def configure_ddp(self):
            self.model = MyCustomDistributedDataParallel(
                self.model,
                device_ids=...,
            )

        def setup(self, trainer):
            # you can access the accelerator and plugins directly
            self.accelerator.setup()
            self.precision_plugin.connect(...)


The custom strategy can then be passed into the ``Trainer`` directly via the ``strategy`` parameter.

.. code-block:: python

    # custom strategy
    trainer = Trainer(strategy=CustomDDPStrategy())


Since the strategy also hosts the Accelerator and various plugins, you can customize all of them to work together as you like:

.. code-block:: python

    # custom strategy, with new accelerator and plugins
    accelerator = MyAccelerator()
    precision_plugin = MyPrecisionPlugin()
    strategy = CustomDDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=strategy)
