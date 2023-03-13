.. include:: links.rst

accelerators
------------

.. currentmodule:: lightning.pytorch.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    CUDAAccelerator
    HPUAccelerator
    IPUAccelerator
    TPUAccelerator

callbacks
---------

.. currentmodule:: lightning.pytorch.callbacks

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    BackboneFinetuning
    BaseFinetuning
    BasePredictionWriter
    BatchSizeFinder
    Callback
    DeviceStatsMonitor
    EarlyStopping
    GradientAccumulationScheduler
    LambdaCallback
    LearningRateFinder
    LearningRateMonitor
    ModelCheckpoint
    ModelPruning
    ModelSummary
    OnExceptionCheckpoint
    ProgressBarBase
    RichModelSummary
    RichProgressBar
    StochasticWeightAveraging
    Timer
    TQDMProgressBar

cli
-----

.. currentmodule:: lightning.pytorch.cli

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LightningCLI
    LightningArgumentParser
    SaveConfigCallback

core
----

.. currentmodule:: lightning.pytorch.core

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~hooks.CheckpointHooks
    ~hooks.DataHooks
    ~hooks.ModelHooks
    LightningDataModule
    LightningModule
    ~mixins.HyperparametersMixin
    ~optimizer.LightningOptimizer


loggers
-------

.. currentmodule:: lightning.pytorch.loggers

.. autosummary::
    :toctree: api
    :nosignatures:

    logger
    comet
    csv_logs
    mlflow
    neptune
    tensorboard
    wandb

plugins
^^^^^^^

precision
"""""""""

.. currentmodule:: lightning.pytorch.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    DeepSpeedPrecisionPlugin
    DoublePrecisionPlugin
    FSDPMixedPrecisionPlugin
    HPUPrecisionPlugin
    IPUPrecisionPlugin
    MixedPrecisionPlugin
    PrecisionPlugin
    TPUBf16PrecisionPlugin
    TPUPrecisionPlugin

environments
""""""""""""

.. currentmodule:: lightning.pytorch.plugins.environments

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    KubeflowEnvironment
    LightningEnvironment
    LSFEnvironment
    MPIEnvironment
    SLURMEnvironment
    TorchElasticEnvironment
    XLAEnvironment

io
""

.. currentmodule:: lightning.pytorch.plugins.io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    AsyncCheckpointIO
    CheckpointIO
    HPUCheckpointIO
    TorchCheckpointIO
    XLACheckpointIO


others
""""""

.. currentmodule:: lightning.pytorch.plugins

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LayerSync
    TorchSyncBatchNorm

profiler
--------

.. currentmodule:: lightning.pytorch.profilers

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

.. currentmodule:: lightning.pytorch.trainer.trainer

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Trainer

strategies
----------

.. currentmodule:: lightning.pytorch.strategies

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    DDPStrategy
    DeepSpeedStrategy
    FSDPStrategy
    HPUParallelStrategy
    IPUStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleHPUStrategy
    SingleTPUStrategy
    Strategy
    XLAStrategy

tuner
-----

.. currentmodule:: lightning.pytorch.tuner.tuning

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Tuner

utilities
---------

.. currentmodule:: lightning.pytorch.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    combined_loader
    data
    deepspeed
    distributed
    memory
    model_summary
    parsing
    rank_zero
    seed
    warnings
