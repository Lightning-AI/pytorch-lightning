###############################
Track and Visualize Experiments
###############################

*******************************
Why do I need to track metrics?
*******************************

In model development, we track values of interest, such as the *validation_loss* to visualize the learning process for our models.
Model development is like driving a car without windows. Charts and logs provide the *windows* to know where to drive the car.

With Lightning, you can visualize virtually anything you can think of: numbers, text, images, and audio.

----

*************
Track metrics
*************

Metric visualization is the most basic but powerful way to understand how your model is doing throughout development.
To track a metric, add the following:

**Step 1:** Pick a logger.

.. code-block:: python

    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger

    # Pick a logger and add it to Fabric
    logger = TensorBoardLogger(root_dir="logs")
    fabric = Fabric(loggers=logger)


Loggers you can choose from:

- :class:`~lightning.fabric.loggers.TensorBoardLogger`
- :class:`~lightning.fabric.loggers.CSVLogger`
- :doc:`WandbLogger <loggers/wandb>`

|

**Step 2:** Add :meth:`~lightning.fabric.fabric.Fabric.log` calls in your code.

.. code-block:: python

    value = ...  # Python scalar or tensor scalar
    fabric.log("some_value", value)


To log multiple metrics at once, use :meth:`~lightning.fabric.fabric.Fabric.log_dict`:

.. code-block:: python

    values = {"loss": loss, "acc": acc, "other": other}
    fabric.log_dict(values)


----


*******************
View logs dashboard
*******************

How you can view the metrics depends on the individual logger you choose.
Most have a dashboard that lets you browse everything you log in real time.

For the :class:`~lightning.fabric.loggers.tensorboard.TensorBoardLogger` shown above, you can open it by running

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

Logging a metric in every iteration can slow down the training.
Reduce the added overhead by logging less frequently:

.. code-block:: python
    :emphasize-lines: 3

    for iteration in range(num_iterations):
        if iteration % log_every_n_steps == 0:
            value = ...
            fabric.log("some_value", value)


----


********************
Use multiple loggers
********************

You can add as many loggers as you want without changing the logging code in your loop.

.. code-block:: python
    :emphasize-lines: 8

    from lightning.fabric import Fabric
    from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

    tb_logger = TensorBoardLogger(root_dir="logs/tensorboard")
    csv_logger = CSVLogger(root_dir="logs/csv")

    # Add multiple loggers in a list
    fabric = Fabric(loggers=[tb_logger, csv_logger])

    # Calling .log() or .log_dict() always logs to all loggers simultaneously
    fabric.log("some_value", value)
