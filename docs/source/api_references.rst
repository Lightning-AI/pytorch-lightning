API References
==============

.. include:: links.rst

Accelerator API
---------------

.. currentmodule:: pytorch_lightning.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    IPUAccelerator
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

Loop API
--------

Base Classes
^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~base.Loop
    ~dataloader.dataloader_loop.DataLoaderLoop


Default Loop Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training
""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    FitLoop
    ~epoch.TrainingEpochLoop
    ~batch.TrainingBatchLoop
    ~optimization.OptimizerLoop
    ~optimization.ManualOptimization


Validation and Testing
""""""""""""""""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~dataloader.EvaluationLoop
    ~epoch.EvaluationEpochLoop


Prediction
""""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~dataloader.PredictionLoop
    ~epoch.PredictionEpochLoop


Plugins API
-----------

Training Type Plugins
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.training_type

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Strategy
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
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    LightningEnvironment
    LSFEnvironment
    TorchElasticEnvironment
    KubeflowEnvironment
    SLURMEnvironment

Checkpoint IO Plugins
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO

Profiler API
------------

.. currentmodule:: pytorch_lightning.profiler

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    AbstractProfiler
    AdvancedProfiler
    BaseProfiler
    PassThroughProfiler
    PyTorchProfiler
    SimpleProfiler
    XLAProfiler


Trainer API
-----------

.. currentmodule:: pytorch_lightning.trainer

.. autosummary::
    :toctree: api
    :nosignatures:

    trainer

LightningLite API
-----------------

.. currentmodule:: pytorch_lightning.lite

.. autosummary::
    :toctree: api
    :nosignatures:

    LightningLite

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
