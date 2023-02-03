:orphan:

.. include:: ../../links.rst

#############
API Reference
#############


Fabric
^^^^^^

.. currentmodule:: lightning_fabric.fabric

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate.rst

    Fabric


Accelerators
^^^^^^^^^^^^

.. currentmodule:: lightning_fabric.accelerators

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    CUDAAccelerator
    MPSAccelerator
    TPUAccelerator


Loggers
^^^^^^^

.. currentmodule:: lightning_fabric.loggers

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate.rst

    Logger
    CSVLogger
    TensorBoardLogger


Plugins
^^^^^^^

Precision
"""""""""

.. TODO(fabric): include DeepSpeedPrecision

.. currentmodule:: lightning_fabric.plugins.precision

.. autosummary::
    :toctree: ../../api
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

.. currentmodule:: lightning_fabric.plugins.environments

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate_noindex.rst

    ~cluster_environment.ClusterEnvironment
    ~kubeflow.KubeflowEnvironment
    ~lightning.LightningEnvironment
    ~lsf.LSFEnvironment
    ~slurm.SLURMEnvironment
    ~torchelastic.TorchElasticEnvironment
    ~xla.XLAEnvironment


IO
""

.. currentmodule:: lightning_fabric.plugins.io

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate.rst

    ~checkpoint_io.CheckpointIO
    ~torch_io.TorchCheckpointIO
    ~xla.XLACheckpointIO


Collectives
"""""""""""

.. currentmodule:: lightning_fabric.plugins.collectives

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate.rst

    Collective
    TorchCollective
    SingleDeviceCollective


Strategies
^^^^^^^^^^

.. TODO(fabric): include DeepSpeedStrategy, XLAStrategy

.. currentmodule:: lightning_fabric.strategies

.. autosummary::
    :toctree: ../../api
    :nosignatures:
    :template: classtemplate.rst

    Strategy
    DDPStrategy
    DataParallelStrategy
    FSDPStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleTPUStrategy
