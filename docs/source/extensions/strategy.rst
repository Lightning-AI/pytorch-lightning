.. _strategy:

########
Strategy
########

Strategy controls the model distribution across training, evaluation, and prediction to be used by the :doc:`Trainer <../common/trainer>`. It can be controlled by passing different
strategy with aliases (``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"`` and so on) as well as a custom strategy to the ``strategy`` parameter for Trainer.

The Strategy in PyTorch Lightning handles the following responsibilities:

* Launch and teardown of training processes (if applicable).
* Setup communication between processes (NCCL, GLOO, MPI, and so on).
* Provide a unified communication interface for reduction, broadcast, and so on.
* Owns the :class:`~pytorch_lightning.core.lightning.LightningModule`
* Handles/owns optimizers and schedulers.


:class:`~pytorch_lightning.strategies.strategy.Strategy` also manages the accelerator, precision, and checkpointing plugins.


****************************************
Training Strategies with Various Configs
****************************************

.. code-block:: python

    # Training with the DistributedDataParallel strategy on 4 GPUs
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with the custom DistributedDataParallel strategy on 4 GPUs
    trainer = Trainer(strategy=DDPStrategy(...), accelerator="gpu", devices=4)

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


----------


************************
Create a Custom Strategy
************************

Expert users may choose to extend an existing strategy by overriding its methods.

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy


    class CustomDDPStrategy(DDPStrategy):
        def configure_ddp(self):
            self.model = MyCustomDistributedDataParallel(
                self.model,
                device_ids=...,
            )

or by subclassing the base class :class:`~pytorch_lightning.strategies.Strategy` to create new ones. These custom strategies
can then be passed into the ``Trainer`` directly via the ``strategy`` parameter.

.. code-block:: python

    # custom plugins
    trainer = Trainer(strategy=CustomDDPStrategy())

    # fully custom accelerator and plugins
    accelerator = MyAccelerator()
    precision_plugin = MyPrecisionPlugin()
    training_strategy = CustomDDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=training_strategy)


The complete list of built-in strategies is listed below.

----------


****************************
Built-In Training Strategies
****************************

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
    IPUStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleTPUStrategy
    Strategy
    TPUSpawnStrategy
