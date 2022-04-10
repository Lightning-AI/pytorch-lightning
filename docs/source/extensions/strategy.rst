.. _strategy:

###################
What is a Strategy?
###################

A Strategy controls the model distribution across training, evaluation, and prediction to be used by the :doc:`Trainer <../common/trainer>`.
It can be controlled by passing different strategy with aliases (``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"`` and so on) as well as a custom strategy object to the ``strategy`` parameter for Trainer.

The Strategy in PyTorch Lightning handles the following responsibilities:

* Launch and teardown of training processes (if applicable).
* Setup communication between processes (NCCL, GLOO, MPI, and so on).
* Provide a unified communication interface for reduction, broadcast, and so on.
* Owns the :class:`~pytorch_lightning.core.lightning.LightningModule`
* Handles/owns optimizers and schedulers.


A Strategy is a composition of one :doc:`Accelerator <../extensions/accelerator>`, one :ref:`Precision Plugin <extensions/plugins:Precision Plugins>`, a :ref:`CheckpointIO <extensions/plugins:CheckpointIO Plugins>` plugin and other optional plugins such as the :ref:`ClusterEnvironment <extensions/plugins:Cluster Environments>`.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/overview.jpeg
    :alt: Illustration of the Strategy as a composition of the Accelerator and several plugins

We expose Strategies mainly for expert users that want to extend Lightning for new hardware support or new distributed backends (e.g. a backend not yet supported by `PyTorch <https://pytorch.org/docs/stable/distributed.html#backends>`_ itself).


----

###########################
Enable Different Strategies
###########################

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

----

########################
Create a Custom Strategy
########################

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
    training_strategy = CustomDDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=training_strategy)


The complete list of built-in strategies is listed below.

----

#############################
Available Training Strategies
#############################

.. currentmodule:: pytorch_lightning.strategies

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    BaguaStrategy
    DDP2Strategy
    DDPFullyShardedStrategy
    DDPShardedStrategy
    DDPSpawnShardedStrategy
    DDPSpawnStrategy
    DDPStrategy
    DataParallelStrategy
    DeepSpeedStrategy
    HorovodStrategy
    HPUParallelStrategy
    IPUStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleHPUStrategy
    SingleTPUStrategy
    Strategy
    TPUSpawnStrategy
