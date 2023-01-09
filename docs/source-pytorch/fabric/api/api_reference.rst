:orphan:

#############
API Reference
#############


Fabric
^^^^^^

.. currentmodule:: lightning_fabric.fabric

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Fabric


Accelerators
^^^^^^^^^^^^

.. currentmodule:: lightning_fabric.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    CUDAAccelerator
    MPSAccelerator
    TPUAccelerator


Plugins
^^^^^^^

Precision
"""""""""

.. currentmodule:: lightning_fabric.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Precision
    DeepSpeedPrecision
    DoublePrecision
    MixedPrecision
    TPUPrecision
    TPUBf16Precision
    FSDPPrecision


Environments
""""""""""""

.. currentmodule:: lightning_fabric.plugins.environments

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
    XLAEnvironment


IO
""

.. currentmodule:: lightning_fabric.plugins.io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO


Collectives
"""""""""""

.. currentmodule:: lightning_fabric.plugins.collectives

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Collective
    TorchCollective
    SingleDeviceCollective


Strategies
^^^^^^^^^^

.. currentmodule:: lightning_fabric.strategies

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Strategy
    DDPStrategy
    DeepSpeedStrategy
    DataParallelStrategy
    DDPShardedStrategy
    FSDPStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleTPUStrategy
    XLAStrategy


Utilities
^^^^^^^^^

.. currentmodule:: pytorch_lightning.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    apply_func
    memory
    rank_zero
    seed
