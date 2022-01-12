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
    GPUAccelerator
    IPUAccelerator
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

Strategy API
------------

.. currentmodule:: pytorch_lightning.strategies

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

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

    ~dataloader.dataloader_loop.DataLoaderLoop
    ~base.Loop


Default Loop Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training
""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~batch.TrainingBatchLoop
    ~epoch.TrainingEpochLoop
    FitLoop
    ~optimization.ManualOptimization
    ~optimization.OptimizerLoop


Validation and Testing
""""""""""""""""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~epoch.EvaluationEpochLoop
    ~dataloader.EvaluationLoop


Prediction
""""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~epoch.PredictionEpochLoop
    ~dataloader.PredictionLoop


Plugins API
-----------

Precision Plugins
^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ApexMixedPrecisionPlugin
    DeepSpeedPrecisionPlugin
    DoublePrecisionPlugin
    FullyShardedNativeMixedPrecisionPlugin
    IPUPrecisionPlugin
    MixedPrecisionPlugin
    NativeMixedPrecisionPlugin
    PrecisionPlugin
    ShardedNativeMixedPrecisionPlugin
    TPUBf16PrecisionPlugin
    TPUPrecisionPlugin

Cluster Environments
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    KubeflowEnvironment
    LightningEnvironment
    LSFEnvironment
    SLURMEnvironment
    TorchElasticEnvironment

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

.. currentmodule:: pytorch_lightning.trainer.trainer

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Trainer

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
