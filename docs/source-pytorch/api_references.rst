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
    CUDAAccelerator
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
    QuantizationAwareTraining
    RichModelSummary
    RichProgressBar
    StochasticWeightAveraging
    Timer
    TQDMProgressBar

cli
-----

.. currentmodule:: pytorch_lightning.cli

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LightningCLI
    LightningArgumentParser
    SaveConfigCallback

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
    ~mixins.HyperparametersMixin
    ~optimizer.LightningOptimizer
    ~saving.ModelIO


loggers
-------

.. currentmodule:: pytorch_lightning.loggers

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

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ColossalAIPrecisionPlugin
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

.. currentmodule:: pytorch_lightning.plugins.environments

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

.. currentmodule:: pytorch_lightning.plugins.io

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

.. currentmodule:: pytorch_lightning.plugins

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LayerSync
    TorchSyncBatchNorm

profiler
--------

.. currentmodule:: pytorch_lightning.profilers

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
    ColossalAIStrategy
    DDPSpawnStrategy
    DDPStrategy
    DataParallelStrategy
    DeepSpeedStrategy
    FSDPStrategy
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

    argparse
    data
    deepspeed
    distributed
    memory
    model_summary
    parsing
    rank_zero
    seed
    warnings
