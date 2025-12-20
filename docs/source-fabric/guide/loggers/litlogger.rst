##########
LitLogger
##########

`LitLogger <https://pypi.org/project/litlogger/>`_ enables seamless experiment tracking, logging, and artifact management on the `Lightning.ai <https://lightning.ai>`_ platform.

It integrates with your Fabric training loop to log metrics, hyperparameters, and model checkpoints automatically to the Lightning Experiments dashboard.
View your experiments at `lightning.ai <https://lightning.ai>`_ with real-time charts, compare runs, and share results with your team.


----


*****************
Set Up LitLogger
*****************

First, install the ``litlogger`` package:

.. code-block:: bash

    pip install litlogger

That's it! LitLogger automatically detects your Lightning.ai credentials when running in a Lightning Studio or when logged in via the CLI.


----


*************
Track Metrics
*************

To start tracking metrics in your training loop, import the LitLogger and configure it with your settings:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    from lightning.fabric import Fabric
    from lightning.pytorch.loggers import LitLogger

    # 1. Configure the logger
    logger = LitLogger(name="my-experiment")

    # 2. Pass it to Fabric
    fabric = Fabric(loggers=logger)


Next, add :meth:`~lightning.fabric.fabric.Fabric.log` calls in your code:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    value = 0.5  # Python scalar or tensor scalar
    fabric.log("some_value", value)


To log multiple metrics at once, use :meth:`~lightning.fabric.fabric.Fabric.log_dict`:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    loss, acc, other = 0.1, 0.95, 0.5
    values = {"loss": loss, "acc": acc, "other": other}
    fabric.log_dict(values)


----


********************
Log Hyperparameters
********************

Log your model's hyperparameters to keep track of your experiment configuration:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    from lightning.pytorch.loggers import LitLogger

    logger = LitLogger(name="my-experiment")
    logger.log_hyperparams({
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "resnet50",
    })


You can also pass metadata directly when creating the logger:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    from lightning.pytorch.loggers import LitLogger

    logger = LitLogger(
        name="my-experiment",
        metadata={"learning_rate": "0.001", "batch_size": "32"},
    )


----


***************
Log Checkpoints
***************

Enable automatic checkpoint logging with the ``log_model`` parameter:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    from lightning.pytorch.loggers import LitLogger

    logger = LitLogger(name="my-experiment", log_model=True)

Checkpoints will be automatically uploaded to the Lightning platform when saved.

You can also manually log model artifacts:

.. code-block:: python

    # Log a model checkpoint file
    logger.log_model_artifact("/path/to/checkpoint.ckpt")

    # Log a model object directly
    logger.log_model(model)


----


*************
Log Files
*************

Log any file as an artifact:

.. code-block:: python

    # Log a configuration file
    logger.log_file("config.yaml")


----


**************************
Capture Terminal Output
**************************

Enable terminal log capture to save your script's output:

.. testcode::
    :skipif: not _LITLOGGER_AVAILABLE

    from lightning.pytorch.loggers import LitLogger

    logger = LitLogger(name="my-experiment", save_logs=True)

Your terminal output will be captured and available in the Lightning Experiments dashboard.


----


*********************
View Your Experiments
*********************

After running your training script, view your experiments at `lightning.ai <https://lightning.ai>`_.
The dashboard provides:

- Real-time metric charts
- Hyperparameter comparison
- Artifact management
- Team collaboration features

Access your experiment URL programmatically:

.. code-block:: python

    print(logger.url)


|
|
