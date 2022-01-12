.. _strategy:

########
Strategy
########

Strategy depicts the training strategy to be used by the :doc:`Lightning Trainer <../common/trainer>`. It can be controlled by passing different
training strategies with aliases (``ddp``, ``ddp_spawn``, ``deepspeed``, etc) as well as custom training strategies to the ``strategy`` parameter for Trainer.

****************************************
Training Strategies with various configs
****************************************

.. code-block:: python

    # Training with the DistributedDataParallel strategy on 4 gpus
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with the custom DistributedDataParallel strategy on 4 gpus
    trainer = Trainer(strategy=DDPPlugin(...), accelerator="gpu", devices=4)

    # Training with the DDP Spawn strategy using auto accelerator selection
    trainer = Trainer(strategy="ddp_spawn", accelerator="auto", devices=4)

    # Training with the DeepSpeed strategy on available gpus
    trainer = Trainer(strategy="deepspeed", accelerator="gpu", devices="auto")

    # Training with the DDP strategy using 3 cpu processes
    trainer = Trainer(strategy="ddp", accelerator="cpu", devices=3)

    # Training with the DDP Spawn strategy on 8 tpu cores
    trainer = Trainer(strategy="ddp_spawn", accelerator="tpu", devices=8)

    # Training with the default IPU strategy on 8 ipus
    trainer = Trainer(accelerator="ipu", devices=8)


Strategy in Lightning handles some of the following responsibilities:

* Launching and teardown of training processes (if applicable)

* Setup communication between processes (NCCL, GLOO, MPI, …)

* Provide a unified communication interface for reduction, broadcast, etc.

* Provide access to the wrapped LightningModule


:class:`~pytorch_lightning.strategies.strategy.Strategy` also manages the accelerator, precision and the checkpointing plugins.


************************
Create a custom Strategy
************************

Expert users may choose to extend an existing strategy by overriding its methods ...

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy


    class CustomDDPStrategy(DDPStrategy):
        def configure_ddp(self):
            self._model = MyCustomDistributedDataParallel(
                self.model,
                device_ids=...,
            )

or by subclassing the base classes :class:`~pytorch_lightning.strategies.Strategy` to create new ones. These custom strategies
can then be passed into the Trainer directly via the ``strategy`` parameter.

.. code-block:: python

    # custom plugins
    trainer = Trainer(strategy=CustomDDPStrategy())

    # fully custom accelerator and plugins
    accelerator = MyAccelerator()
    precision_plugin = MyPrecisionPlugin()
    training_strategy = CustomDDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=training_strategy)


The full list of built-in strategies is listed below.

----------


Training Strategies
-------------------

.. currentmodule:: pytorch_lightning.strategies

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    Strategy
    SingleDeviceStrategy
    ParallelStrategy
    DataParallelStrategy
    DDPStrategy
    DDP2Strategy
    DDPShardedStrategy
    DDPSpawnShardedStrategy
    DDPSpawnStrategy
    DeepSpeedStrategy
    HorovodStrategy
    SingleTPUStrategy
    TPUSpawnStrategy
