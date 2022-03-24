.. _plugins:

#######
Plugins
#######

.. include:: ../links.rst

Plugins allow custom integrations to the internals of the Trainer such as a custom precision, checkpointing or
cluster environment implementation.

Under the hood, the Lightning Trainer is using plugins in the training routine, added automatically
depending on the provided Trainer arguments. For example:

.. code-block:: python

    # accelerator: GPUAccelerator
    # training strategy: DDPStrategy
    # precision: NativeMixedPrecisionPlugin
    trainer = Trainer(accelerator="gpu", devices=4, precision=16)


We expose Accelerators and Plugins mainly for expert users that want to extend Lightning for:

- New hardware (like TPU plugin)
- Distributed backends (e.g. a backend not yet supported by
  `PyTorch <https://pytorch.org/docs/stable/distributed.html#backends>`_ itself)
- Clusters (e.g. customized access to the cluster's environment interface)

There are three types of Plugins in Lightning with different responsibilities:

Precision Plugins
-----------------

We expose precision plugins for the users


.. code-block:: python

    # precision: FP16Plugin
    trainer = Trainer(precision=16)

- Precision Plugins
- CheckpointIO Plugins
- Cluster Environments (e.g. customized access to the cluster's environment interface)


The full list of built-in plugins is listed below.


.. warning:: The Plugin API is in beta and subject to change.
    For help setting up custom plugins/accelerators, please reach out to us at **support@pytorchlightning.ai**


Precision Plugins
-----------------

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
