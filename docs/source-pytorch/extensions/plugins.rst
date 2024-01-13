.. _plugins:

#######
Plugins
#######

.. include:: ../links.rst

Plugins allow custom integrations to the internals of the Trainer such as custom precision, checkpointing or
cluster environment implementation.

Under the hood, the Lightning Trainer is using plugins in the training routine, added automatically
depending on the provided Trainer arguments.

There are three types of plugins in Lightning with different responsibilities:

- Precision plugins
- CheckpointIO plugins
- Cluster environments

You can make the Trainer use one or multiple plugins by adding it to the ``plugins`` argument like so:

.. code-block:: python

    trainer = Trainer(plugins=[plugin1, plugin2, ...])


By default, the plugins get selected based on the rest of the Trainer settings such as the ``strategy``.


-----------

.. _precision-plugins:

*****************
Precision Plugins
*****************

We provide precision plugins for you to benefit from numerical representations with lower precision than
32-bit floating-point or higher precision, such as 64-bit floating-point.

.. code-block:: python

    # Training with 16-bit precision
    trainer = Trainer(precision=16)

The full list of built-in precision plugins is listed below.

.. currentmodule:: lightning.pytorch.plugins.precision

.. autosummary::
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

More information regarding precision with Lightning can be found :ref:`here <precision>`

-----------


.. _checkpoint_io_plugins:

********************
CheckpointIO Plugins
********************

As part of our commitment to extensibility, we have abstracted Lightning's checkpointing logic into the :class:`~lightning.pytorch.plugins.io.CheckpointIO` plugin.
With this, you have the ability to customize the checkpointing logic to match the needs of your infrastructure.

Below is a list of built-in plugins for checkpointing.

.. currentmodule:: lightning.pytorch.plugins.io

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    AsyncCheckpointIO
    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO

Learn more about custom checkpointing with Lightning :ref:`here <checkpointing_expert>`.

-----------


.. _cluster_environment_plugins:

********************
Cluster Environments
********************

You can define the interface of your own cluster environment based on the requirements of your infrastructure.

.. currentmodule:: lightning.pytorch.plugins.environments

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    KubeflowEnvironment
    LightningEnvironment
    LSFEnvironment
    SLURMEnvironment
    TorchElasticEnvironment
    XLAEnvironment
