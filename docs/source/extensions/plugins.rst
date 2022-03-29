.. _plugins:

#######
Plugins
#######

.. include:: ../links.rst

Plugins allow custom integrations to the internals of the Trainer such as a custom precision, checkpointing or
cluster environment implementation.

Under the hood, the Lightning Trainer is using plugins in the training routine, added automatically
depending on the provided Trainer arguments.

There are three types of Plugins in Lightning with different responsibilities:

- Precision Plugins
- CheckpointIO Plugins
- Cluster Environments (e.g. customized access to the cluster's environment interface)


Precision Plugins
-----------------

We provide precision plugins for the users so that they can benefit from numerical representations with lower precision than
32-bit floating-point or higher precision, such as 64-bit floating-point.

.. code-block:: python

    # precision: FP16Plugin
    trainer = Trainer(precision=16)

The full list of built-in precision plugins is listed below.

.. currentmodule:: pytorch_lightning.plugins.precision

.. autosummary::
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


CheckpointIO Plugins
--------------------

As part of our commitment to extensibility, we have abstracted Lightning's checkpointing logic into the :class:`~pytorch_lightning.plugins.io.CheckpointIO` plugin.
With this, users have the ability to customize the checkpointing logic to match the needs of their infrastructure.

Below is a list of built-in plugins for checkpointing.

.. currentmodule:: pytorch_lightning.plugins.io

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO

You could learn more about custom checkpointing with Lightning :ref:`here <../common/checkpointing:Customize Checkpointing>`.

Cluster Environments
--------------------

Clusters (e.g. customized access to the cluster's environment interface)

.. currentmodule:: pytorch_lightning.plugins.environments

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    LightningEnvironment
    LSFEnvironment
    TorchElasticEnvironment
    KubeflowEnvironment
    SLURMEnvironment
