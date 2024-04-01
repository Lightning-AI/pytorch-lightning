.. include:: links.rst

accelerators
------------

.. currentmodule:: lightning_pytorch.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    CUDAAccelerator
    XLAAccelerator

callbacks
---------

.. currentmodule:: lightning_pytorch.callbacks

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
    ProgressBar
    RichModelSummary
    RichProgressBar
    StochasticWeightAveraging
    SpikeDetection
    ThroughputMonitor
    Timer
    TQDMProgressBar

cli
-----

.. currentmodule:: lightning_pytorch.cli

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LightningCLI
    LightningArgumentParser
    SaveConfigCallback

core
----

.. currentmodule:: lightning_pytorch.core

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


.. _loggers-api-references:

loggers
-------

.. currentmodule:: lightning_pytorch.loggers

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

.. currentmodule:: lightning_pytorch.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    DeepSpeedPrecision
    DoublePrecision
    HalfPrecision
    FSDPPrecision
    MixedPrecision
    Precision
    XLAPrecision
    TransformerEnginePrecision
    BitsandbytesPrecision

environments
""""""""""""

.. currentmodule:: lightning_pytorch.plugins.environments

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

.. currentmodule:: lightning_pytorch.plugins.io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    AsyncCheckpointIO
    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO


others
""""""

.. currentmodule:: lightning_pytorch.plugins

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    LayerSync
    TorchSyncBatchNorm

profiler
--------

.. currentmodule:: lightning_pytorch.profilers

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

.. currentmodule:: lightning_pytorch.trainer.trainer

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Trainer

strategies
----------

.. currentmodule:: lightning_pytorch.strategies

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    DDPStrategy
    DeepSpeedStrategy
    FSDPStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleDeviceXLAStrategy
    Strategy
    XLAStrategy

tuner
-----

.. currentmodule:: lightning_pytorch.tuner.tuning

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Tuner

utilities
---------

.. currentmodule:: lightning_pytorch.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    combined_loader
    data
    deepspeed
    memory
    model_summary
    parsing
    rank_zero
    seed
    warnings

.. autofunction:: lightning_pytorch.utilities.measure_flops
