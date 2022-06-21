.. include:: links.rst

accelerators
------------

.. currentmodule:: pytorch_lightning.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    GPUAccelerator
    HPUAccelerator
    IPUAccelerator
    TPUAccelerator

callbacks
---------

.. currentmodule:: pytorch_lightning.callbacks

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    BackboneFinetuning
    BaseFinetuning
    BasePredictionWriter
    Callback
    DeviceStatsMonitor
    EarlyStopping
    GradientAccumulationScheduler
    LambdaCallback
    LearningRateMonitor
    ModelCheckpoint
    ModelPruning
    ModelSummary
    ProgressBarBase
    QuantizationAwareTraining
    RichModelSummary
    RichProgressBar
    StochasticWeightAveraging
    Timer
    TQDMProgressBar

core
----

.. currentmodule:: pytorch_lightning.core

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~hooks.CheckpointHooks
    ~hooks.DataHooks
    ~hooks.ModelHooks
    LightningDataModule
    LightningModule
    ~mixins.DeviceDtypeModuleMixin
    ~mixins.HyperparametersMixin
    ~optimizer.LightningOptimizer
    ~saving.ModelIO

lightninglite
-------------

.. currentmodule:: pytorch_lightning.lite

.. autosummary::
    :toctree: api
    :nosignatures:

    LightningLite

loggers
-------

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
    wandb

loops
^^^^^

Base Classes
""""""""""""

.. currentmodule:: pytorch_lightning.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~dataloader.dataloader_loop.DataLoaderLoop
    ~base.Loop

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


plugins
^^^^^^^

precision
"""""""""

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ApexMixedPrecisionPlugin
    DeepSpeedPrecisionPlugin
    DoublePrecisionPlugin
    FullyShardedNativeMixedPrecisionPlugin
    HPUPrecisionPlugin
    IPUPrecisionPlugin
    MixedPrecisionPlugin
    NativeMixedPrecisionPlugin
    PrecisionPlugin
    ShardedNativeMixedPrecisionPlugin
    TPUBf16PrecisionPlugin
    TPUPrecisionPlugin

environments
""""""""""""

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

io
""

.. currentmodule:: pytorch_lightning.plugins.io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    HPUCheckpointIO
    TorchCheckpointIO
    XLACheckpointIO


others
""""""

.. currentmodule:: pytorch_lightning.plugins

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LayerSync
    NativeSyncBatchNorm

profiler
--------

.. currentmodule:: pytorch_lightning.profiler

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    AdvancedProfiler
    PassThroughProfiler
    Profiler
    PyTorchProfiler
    SimpleProfiler
    XLAProfiler

trainer
-------

.. currentmodule:: pytorch_lightning.trainer.trainer

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Trainer

strategies
----------

.. currentmodule:: pytorch_lightning.strategies

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    BaguaStrategy
    CollaborativeStrategy
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

tuner
-----

.. currentmodule:: pytorch_lightning.tuner.tuning

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Tuner

utilities
---------

.. currentmodule:: pytorch_lightning.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    apply_func
    argparse
    cli
    cloud_io
    deepspeed
    distributed
    finite_checks
    memory
    model_summary
    optimizer
    parsing
    rank_zero
    seed
    warnings
