:orphan:

.. _precision_expert:

########################
N-Bit Precision (Expert)
########################
**Audience:** Researchers looking to integrate their new precision techniques into Lightning.


*****************
Precision Plugins
*****************

You can also customize and pass your own Precision Plugin by subclassing the :class:`~pytorch_lightning.plugins.precision.precision_plugin.PrecisionPlugin` class.

- Perform pre and post backward/optimizer step operations such as scaling gradients.
- Provide context managers for forward, training_step, etc.

.. code-block:: python

    class CustomPrecisionPlugin(PrecisionPlugin):
        precision = 16

        ...


    trainer = Trainer(plugins=[CustomPrecisionPlugin()])
