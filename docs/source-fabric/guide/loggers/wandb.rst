##################
Weights and Biases
##################

TODO: Brief Description here


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
    from wandb.x.y.z import WandbLogger

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

With WandbLogger you can also log images, text, tables, checkpoints, hyperparameters and more.
For a description of all features, check out the official Weights and Biases documentation and examples.


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Official WandbLogger Documentation
    :description: Learn about all features from Weights and Biases
    :button_link: https://docs.wandb.ai/guides/integrations/lightning
    :col_css: col-md-4
    :height: 150

.. displayitem::
    :header: WandbLogger Examples
    :description: See examples of how to use the WandbLogger
    :button_link: https://github.com/wandb/examples
    :col_css: col-md-4
    :height: 150


.. raw:: html

        </div>
    </div>


|
|
