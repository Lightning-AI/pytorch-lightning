.. _plugins:

#######
Plugins
#######

.. include:: ../links.rst

Plugins allow custom integrations to the internals of the Trainer such as a custom precision types, checkpoint IO, or multi-node clusters.

Under the hood, the Lightning Trainer is using plugins in the training routine, added automatically
depending on the provided Trainer arguments. For example:

.. code-block:: python

    # accelerator: GPUAccelerator
    # training strategy: DDPStrategy
    # precision plugin: NativeMixedPrecisionPlugin
    trainer = Trainer(gpus=4, precision=16)

You can force a specific plugin by adding it to the ``plugins`` Trainer argument like so:

.. code-block:: python

    # pass in one or multiple plugins
    trainer = Trainer(plugins=[my_plugin], ...)

The complete list of built-in plugins you can select from is listed below.
There are three categories currently available:

- Precision:
- Checkpoint IO:
- Cluster Environment

For each type there is a base class of the same name that you can extend to create new custom plugins, for example:

.. code-block:: python

    from pytorch_lightning.strategies import DDPStrategy, StrategyRegistry, CheckpointIO

    class MyCheckpointIO(CheckpointIO):

        def save_checkpoint(self, checkpoint):
            ...

        def load_checkpoint(self, path):
            ...


    my_checkpoint_io = MyCheckpointIO()
    trainer = Trainer(plugins=[my_checkpoint_io])



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

Checkpoint IO Plugins
---------------------

.. currentmodule:: pytorch_lightning.plugins.io

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO


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
