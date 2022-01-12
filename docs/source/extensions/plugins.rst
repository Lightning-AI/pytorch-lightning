.. _plugins:

#######
Plugins
#######

.. include:: ../links.rst

Plugins allow custom integrations to the internals of the Trainer such as a custom precision or
distributed implementation.

Under the hood, the Lightning Trainer is using plugins in the training routine, added automatically
depending on the provided Trainer arguments. For example:

.. code-block:: python

    # accelerator: GPUAccelerator
    # training strategy: DDPStrategy
    # precision: NativeMixedPrecisionPlugin
    trainer = Trainer(gpus=4, precision=16)


We expose Accelerators and Plugins mainly for expert users that want to extend Lightning for:

- New hardware (like TPU plugin)
- Distributed backends (e.g. a backend not yet supported by
  `PyTorch <https://pytorch.org/docs/stable/distributed.html#backends>`_ itself)
- Clusters (e.g. customized access to the cluster's environment interface)

There are two types of Plugins in Lightning with different responsibilities:

Strategy
--------

- Launching and teardown of training processes (if applicable)
- Setup communication between processes (NCCL, GLOO, MPI, ...)
- Provide a unified communication interface for reduction, broadcast, etc.
- Provide access to the wrapped LightningModule


Futhermore, for multi-node training Lightning provides cluster environment plugins that allow the advanced user
to configure Lightning to integrate with a :ref:`custom-cluster`.


.. image:: ../_static/images/accelerator/overview.svg


**********************
Create a custom plugin
**********************

Expert users may choose to extend an existing plugin by overriding its methods ...

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy


    class CustomDDPStrategy(DDPStrategy):
        def configure_ddp(self):
            self._model = MyCustomDistributedDataParallel(
                self.model,
                device_ids=...,
            )

or by subclassing the base classes :class:`~pytorch_lightning.strategies.Strategy` or
:class:`~pytorch_lightning.plugins.precision.PrecisionPlugin` to create new ones. These custom plugins
can then be passed into the Trainer directly or via a (custom) accelerator:

.. code-block:: python

    # custom plugins
    trainer = Trainer(strategy=CustomDDPStrategy(), plugins=[CustomPrecisionPlugin()])

    # fully custom accelerator and plugins
    accelerator = MyAccelerator()
    precision_plugin = MyPrecisionPlugin()
    training_type_plugin = CustomDDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=training_type_plugin)


The full list of built-in plugins is listed below.


.. warning:: The Plugin API is in beta and subject to change.
    For help setting up custom plugins/accelerators, please reach out to us at **support@pytorchlightning.ai**


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



Precision Plugins
-----------------

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    PrecisionPlugin
    MixedPrecisionPlugin
    NativeMixedPrecisionPlugin
    ShardedNativeMixedPrecisionPlugin
    ApexMixedPrecisionPlugin
    DeepSpeedPrecisionPlugin
    TPUPrecisionPlugin
    TPUBf16PrecisionPlugin
    DoublePrecisionPlugin
    FullyShardedNativeMixedPrecisionPlugin
    IPUPrecisionPlugin


Cluster Environments
--------------------

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    LightningEnvironment
    LSFEnvironment
    TorchElasticEnvironment
    KubeflowEnvironment
    SLURMEnvironment
