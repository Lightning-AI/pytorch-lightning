.. include:: links.rst

#############
API Reference
#############


Fabric
^^^^^^

.. currentmodule:: lightning.fabric.fabric

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    Fabric


Accelerators
^^^^^^^^^^^^

.. currentmodule:: lightning.fabric.accelerators

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    CUDAAccelerator
    MPSAccelerator
    TPUAccelerator


Loggers
^^^^^^^

.. currentmodule:: lightning.fabric.loggers

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    Logger
    CSVLogger
    TensorBoardLogger

Precision
"""""""""

.. TODO(fabric): include DeepSpeedPrecision

.. currentmodule:: lightning.fabric.plugins.precision

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    Precision
    DoublePrecision
    MixedPrecision
    TPUPrecision
    TPUBf16Precision
    FSDPPrecision


Environments
""""""""""""

.. currentmodule:: lightning.fabric.plugins.environments

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate_noindex.rst

    ~cluster_environment.ClusterEnvironment
    ~kubeflow.KubeflowEnvironment
    ~lightning.LightningEnvironment
    ~lsf.LSFEnvironment
    ~mpi.MPIEnvironment
    ~slurm.SLURMEnvironment
    ~torchelastic.TorchElasticEnvironment
    ~xla.XLAEnvironment


IO
""

.. currentmodule:: lightning.fabric.plugins.io

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    ~checkpoint_io.CheckpointIO
    ~torch_io.TorchCheckpointIO
    ~xla.XLACheckpointIO


Collectives
"""""""""""

.. currentmodule:: lightning.fabric.plugins.collectives

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    Collective
    TorchCollective
    SingleDeviceCollective


Strategies
^^^^^^^^^^

.. TODO(fabric): include DeepSpeedStrategy, XLAStrategy

.. currentmodule:: lightning.fabric.strategies

.. autosummary::
    :toctree: api/generated
    :nosignatures:
    :template: classtemplate.rst

    Strategy
    DDPStrategy
    DataParallelStrategy
    FSDPStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleTPUStrategy
