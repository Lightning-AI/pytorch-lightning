API References
==============

Accelerator API
---------------

.. currentmodule:: pytorch_lightning.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    GPUAccelerator
    TPUAccelerator

Core API
--------

.. currentmodule:: pytorch_lightning.core

.. autosummary::
    :toctree: api
    :nosignatures:

    datamodule
    decorators
    hooks
    lightning

Callbacks API
-------------

.. currentmodule:: pytorch_lightning.callbacks

.. autosummary::
    :toctree: api
    :nosignatures:

    base
    early_stopping
    gpu_stats_monitor
    gradient_accumulation_scheduler
    lr_monitor
    model_checkpoint
    progress

Loggers API
-----------

.. currentmodule:: pytorch_lightning.loggers

.. autosummary::
    :toctree: api
    :nosignatures:

    base
    comet
    csv_logs
    mlflow
    neptune
    tensorboard
    test_tube
    wandb

Plugins API
-----------

Training Type Plugins
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.training_type

.. autosummary::
    :toctree: api
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
^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
    :toctree: api
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
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    LightningEnvironment
    TorchElasticEnvironment
    KubeflowEnvironment
    SLURMEnvironment


Profiler API
------------

.. currentmodule:: pytorch_lightning.profiler

.. autosummary::
    :toctree: api
    :nosignatures:

    profilers

Trainer API
-----------

.. currentmodule:: pytorch_lightning.trainer

.. autosummary::
    :toctree: api
    :nosignatures:

    trainer

Tuner API
---------

.. currentmodule:: pytorch_lightning.tuner.tuning

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Tuner

Utilities API
-------------

.. currentmodule:: pytorch_lightning.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    cli
    argparse
    seed
