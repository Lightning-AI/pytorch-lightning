.. _plugins:

#######
Plugins
#######

.. include:: ../links.rst

Plugins allow custom integrations to the internals of the Trainer such as custom precision, checkpointing or
cluster environment implementation.

Under the hood, the Lightning Trainer is using plugins in the training routine, added automatically
depending on the provided Trainer arguments.

There are three types of Plugins in Lightning with different responsibilities:

- Precision Plugins
- CheckpointIO Plugins
- Cluster Environments


*****************
Precision Plugins
*****************

We provide precision plugins for you to benefit from numerical representations with lower precision than
32-bit floating-point or higher precision, such as 64-bit floating-point.

.. code-block:: python

    # Training with 16-bit precision
    trainer = Trainer(precision=16)

The full list of built-in precision plugins is listed below.

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
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

More information regarding precision with Lightning can be found :doc:`here <../advanced/precision>`

-----------

********************
CheckpointIO Plugins
********************

As part of our commitment to extensibility, we have abstracted Lightning's checkpointing logic into the :class:`~pytorch_lightning.plugins.io.CheckpointIO` plugin.
With this, you have the ability to customize the checkpointing logic to match the needs of your infrastructure.

Below is a list of built-in plugins for checkpointing.

.. currentmodule:: pytorch_lightning.plugins.io

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    HPUCheckpointIO
    TorchCheckpointIO
    XLACheckpointIO

You could learn more about custom checkpointing with Lightning :ref:`here <customize_checkpointing>`.

-----------

********************
Cluster Environments
********************

You can define the interface of your own cluster environment based on the requirements of your infrastructure.

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    KubeflowEnvironment
    LightningEnvironment
    LSFEnvironment
    SLURMEnvironment
    TorchElasticEnvironment
