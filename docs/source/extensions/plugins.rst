.. _plugins:

#######
Plugins
#######

Plugins allow custom integrations to the internals of the Trainer such as a custom amp or ddp implementation.



.. warning:: The Plugin API is in beta and subject to change.
    For help setting up custom plugins/accelerators, please reach out to us at **support@pytorchlightning.ai**


Training Type Plugins
---------------------

.. currentmodule:: pytorch_lightning.plugins.training_type

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    TrainingTypePlugin
    SingleDevicePlugin
    ParallelPlugin
    DataParallelPlugin
    DDPPlugin
    DDP2Plugin
    DDPShardedPlugin
    DDPSpawnShardedPlugin
    DDPSpawnPlugin
    DeepSpeedPlugin
    HorovodPlugin
    RPCPlugin
    RPCSequentialPlugin
    SingleTPUPlugin
    TPUSpawnPlugin


Precision Plugins
-----------------

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    PrecisionPlugin
    NativeMixedPrecisionPlugin
    ShardedNativeMixedPrecisionPlugin
    ApexMixedPrecisionPlugin
    DeepSpeedPrecisionPlugin
    TPUHalfPrecisionPlugin
    DoublePrecisionPlugin


Cluster Environments
--------------------

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    LightningEnvironment
    TorchElasticEnvironment
    SLURMEnvironment

