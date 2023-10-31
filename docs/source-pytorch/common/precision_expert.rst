:orphan:

.. _precision_expert:

########################
N-Bit Precision (Expert)
########################
**Audience:** Researchers looking to integrate their new precision techniques into Lightning.


*****************
Precision Plugins
*****************

You can also customize and pass your own Precision Plugin by subclassing the :class:`~lightning.pytorch.plugins.precision.precision.Precision` class.

- Perform pre and post backward/optimizer step operations such as scaling gradients.
- Provide context managers for forward, training_step, etc.

.. code-block:: python

    class CustomPrecision(Precision):
        precision = "16-mixed"

        ...


    trainer = Trainer(plugins=[CustomPrecision()])
