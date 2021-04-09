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

.. currentmodule:: pytorch_lightning.plugins

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Plugin
    ApexMixedPrecisionPlugin
    DeepSpeedPrecisionPlugin
    DoublePrecisionPlugin
    NativeMixedPrecisionPlugin
    PrecisionPlugin
    ShardedNativeMixedPrecisionPlugin
    TPUHalfPrecisionPlugin
    DDPPlugin
    DDP2Plugin
    DDPSpawnPlugin
    DeepSpeedPlugin
    DataParallelPlugin
    HorovodPlugin
    ParallelPlugin
    RPCPlugin
    RPCSequentialPlugin
    DDPShardedPlugin
    DDPSpawnShardedPlugin
    SingleDevicePlugin
    SingleTPUPlugin
    TPUSpawnPlugin
    TrainingTypePlugin

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

.. currentmodule:: pytorch_lightning.tuner

.. autosummary::
    :toctree: api
    :nosignatures:

    batch_size_scaling
    lr_finder

Utilities API
-------------

.. currentmodule:: pytorch_lightning.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    cli
    argparse_utils
    seed
