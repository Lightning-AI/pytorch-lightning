##################
VisualDL Logger
##################

`VisualDL <https://www.paddlepaddle.org.cn/paddle/visualdl>`_ is a visualization analysis tool that provides a variety of charts to show the trends of parameters, visualizes model structures, data samples, histograms of tensors, PR curves, ROC curves, and high-dimensional data distributions.

It integrates seamlessly with your Fabric training loop to log metrics, images, audio, histograms, embeddings, and more via the `VisualDLLogger`.


----


*************************
Set Up VisualDL Logger
*************************

First, you need to install the ``visualdl`` package:

.. code-block:: bash

    pip install visualdl


----


*************
Track Metrics
*************

To start tracking metrics in your training loop, import the VisualDLLogger and configure it with your settings:

.. code-block:: python

    from lightning.fabric import Fabric
    from lightning.fabric.loggers import VisualDLLogger

    # 1. Configure the logger
    logger = VisualDLLogger(root_dir="logs", name="my_experiment")

    # 2. Pass it to Fabric
    fabric = Fabric(loggers=logger)

Next, add :meth:`~lightning.fabric.fabric.Fabric.log` calls in your code:

.. code-block:: python

    value = 0.5  # Python scalar or tensor scalar
    fabric.log("some_value", value)

To log multiple metrics at once, use :meth:`~lightning.fabric.fabric.Fabric.log_dict`:

.. code-block:: python

    values = {"loss": loss, "acc": acc, "other": other}
    fabric.log_dict(values)


----


********************
Log Hyperparameters
********************

Log your model's hyperparameters:

.. code-block:: python

    from lightning.fabric.loggers import VisualDLLogger

    logger = VisualDLLogger(root_dir="logs", name="my_experiment")
    logger.log_hyperparams({
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "resnet50",
    })


----


**********************
Log Images, Audio, etc.
**********************

VisualDLLogger supports logging various media types through its experiment interface:

.. code-block:: python

    # Access the underlying VisualDL writer
    visualdl_logger = fabric.logger.experiment

    # Log image
    visualdl_logger.add_image(tag="generated", img=image_tensor, step=global_step)

    # Log audio
    visualdl_logger.add_audio(tag="sample", audio=audio_tensor, step=global_step, sample_rate=16000)

    # Log histogram
    visualdl_logger.add_histogram(tag="weights", values=weights, step=global_step)

    # Log embeddings
    visualdl_logger.add_embeddings(tag="embeddings", mat=features, metadata=labels)

    # Log PR curve
    visualdl_logger.add_pr_curve(tag="pr_curve", labels=labels, predictions=probs, step=global_step)

    # Log ROC curve
    visualdl_logger.add_roc_curve(tag="roc_curve", labels=labels, predictions=probs, step=global_step)


----


*********************
View Your Experiments
*********************

After running your training script, launch the VisualDL dashboard to view your logs:

.. code-block:: bash

    visualdl --logdir ./logs


The dashboard provides:

- Real-time metric charts
- Image and audio visualization
- Histogram analysis
- Embedding projection
- PR/ROC curves
- Model graph visualization (requires manual export)


----


*************************
Advanced Configuration
*************************

The VisualDLLogger accepts several configuration parameters:

.. code-block:: python

    from lightning.fabric.loggers import VisualDLLogger

    logger = VisualDLLogger(
        root_dir="logs",
        name="my_experiment",
        version="experiment_v1",  # Custom version
        sub_dir="subfolder",       # Subdirectory within version
        display_name="My Experiment",  # Display name in VisualDL panel
        file_name="custom_log.log",     # Custom log file name
        max_queue=50,              # Queue size before flushing
        flush_secs=60,             # Flush interval in seconds
    )


----


***********
API Summary
***********

.. autoclass:: lightning.fabric.loggers.visualdl.VisualDLLogger
    :noindex:
    :show-inheritance:
    :members:
    :exclude-members: __init__, __weakref__


|
|
