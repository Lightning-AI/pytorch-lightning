##################
Weights and Biases
##################

`Weights & Biases (W&B) <https://wandb.ai>`_  allows machine learning practitioners to track experiments, visualize data, and share insights with a few lines of code.

It integrates seamlessly with your Lightning ML workflows to log metrics, output visualizations, and manage artifacts.
This integration provides a simple way to log metrics and artifacts from your Fabric training loop to W&B via the ``WandbLogger``.
The ``WandbLogger`` also supports all features of the Weights and Biases library, such as logging rich media (image, audio, video), artifacts, hyperparameters, tables, custom visualizations, and more.
`Check the official documentation here <https://docs.wandb.ai>`_.


----


*************************
Set Up Weights and Biases
*************************

First, you need to install the ``wandb`` package:

.. code-block:: bash

    pip install wandb

Then log in with your API key found in your W&B account settings:

.. code-block:: bash

    wandb login <your-api-key>


You are all set and can start logging your metrics to Weights and Biases.


----


*************
Track metrics
*************

To start tracking metrics in your training loop, import the WandbLogger and configure it with your settings:

.. code-block:: python

    from lightning.fabric import Fabric

    # 1. Import the WandbLogger
    from wandb.integration.lightning.fabric import WandbLogger

    # 2. Configure the logger
    logger = WandbLogger(project="my-project")

    # 3. Pass it to Fabric
    fabric = Fabric(loggers=logger)


Next, add :meth:`~lightning.fabric.fabric.Fabric.log` calls in your code.

.. code-block:: python

    value = ...  # Python scalar or tensor scalar
    fabric.log("some_value", value)


To log multiple metrics at once, use :meth:`~lightning.fabric.fabric.Fabric.log_dict`:

.. code-block:: python

    values = {"loss": loss, "acc": acc, "other": other}
    fabric.log_dict(values)


----


**************************************************
Logging media, artifacts, hyperparameters and more
**************************************************

With ``WandbLogger`` you can also log images, text, tables, checkpoints, hyperparameters and more.
For a description of all features, check out the official Weights and Biases documentation and examples.


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Official WandbLogger Lightning and Fabric Documentation
    :description: Learn about all features from Weights and Biases
    :button_link: https://docs.wandb.ai/guides/integrations/lightning
    :col_css: col-md-4
    :height: 150

.. displayitem::
    :header: Fabric WandbLogger Example
    :description: Official example of how to use the WandbLogger with Fabric
    :button_link: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Track_PyTorch_Lightning_with_Fabric_and_Wandb.ipynb
    :col_css: col-md-4
    :height: 150

.. displayitem::
    :header: Lightning WandbLogger Example
    :description: Official example of how to use the WandbLogger with Lightning
    :button_link: wandb.me/lightning
    :col_css: col-md-4
    :height: 150


.. raw:: html

        </div>
    </div>


|
|
