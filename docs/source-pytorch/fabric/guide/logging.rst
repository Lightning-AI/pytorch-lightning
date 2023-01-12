:orphan:

###############################
Track and Visualize Experiments
###############################

*******************************
Why do I need to track metrics?
*******************************

In model development, we track values of interest such as the *validation_loss* to visualize the learning process for our models.
Model development is like driving a car without windows, charts and logs provide the *windows* to know where to drive the car.

With Lightning, you can visualize virtually anything you can think of: numbers, text, images, audio.

----

*************
Track metrics
*************

Metric visualization is the most basic but powerful way of understanding how your model is doing throughout the model development process.
To track a metric, add the following:

**Step 1:** Pick a logger.

.. code-block:: python

    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger

    # Pick a logger and add it to Fabric
    logger = TensorBoardLogger(root_dir="logs")
    fabric = Fabric(loggers=logger)


Built-in loggers you can choose from:

- :class:`~lightning_fabric.loggers.TensorBoardLogger`
- :class:`~lightning_fabric.loggers.CSVLogger`

|

**Step 2:** Add :meth:`~lightning_fabric.fabric.Fabric.log` calls in your code.

.. code-block:: python

    value = ...  # Python scalar or tensor scalar
    fabric.log("some_value", value)


To log multiple metrics at once, use :meth:`~lightning_fabric.fabric.Fabric.log_dict`:

.. code-block:: python

    values = {"loss": loss, "acc": acc, "other": other}
    fabric.log_dict(values)


----


*******************
View logs dashboard
*******************

How you can view the metrics depends on the individual logger you choose.
Most of them have a dashboard that lets you browse everything you log in real time.

For the :class:`~lightning_fabric.loggers.tensorboard.TensorBoardLogger` shown above, you can open it by running

.. code-block:: bash

    tensorboard --logdir=./logs

If you're using a notebook environment such as *Google Colab* or *Kaggle* or *Jupyter*, launch TensorBoard with this command

.. code-block:: bash

    %reload_ext tensorboard
    %tensorboard --logdir=./logs


----


*************************
Control logging frequency
*************************

Logging a metric in every iteration can slow down training.
Reduce the added overhead by logging less frequently:

.. code-block:: python
    :emphasize-lines: 3

    for iteration in range(num_iterations):

        if iteration % log_every_n_steps == 0:
            value = ...
            fabric.log("some_value", value)
