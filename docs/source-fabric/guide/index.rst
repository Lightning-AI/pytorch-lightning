#############
How-to Guides
#############


******
Basics
******


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Convert to Fabric in 5 minutes
    :description: Learn how to add Fabric to your PyTorch code
    :button_link: ../fundamentals/convert.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Scale your model with Accelerators
    :description: Take advantage of your hardware with a switch of a flag
    :button_link: ../fundamentals/accelerators.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Structure your Fabric code
    :description: Best practices for setting up your training script with Fabric
    :button_link: ../fundamentals/code_structure.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Launch distributed training
    :description: Launch a Python script on multiple devices and machines
    :button_link: ../fundamentals/launch.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Launch Fabric in a notebook
    :description: Launch on multiple devices from within a Jupyter notebook
    :button_link: ../fundamentals/notebooks.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Improve performance with Mixed-Precision training
    :description: Save memory and speed up training using mixed precision
    :button_link: ../fundamentals/precision.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. raw:: html

        </div>
    </div>



**********************
Build your own Trainer
**********************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Organize your model code with with LightningModule
    :description: Organize your code in a LightningModule and use it with Fabric
    :button_link: lightning_module.html
    :col_css: col-md-4
    :height: 170
    :tag: intermediate

.. displayitem::
    :header: Encapsulate code into Callbacks
    :description: Make use of the Callback system in Fabric
    :button_link: callbacks.html
    :col_css: col-md-4
    :height: 170
    :tag: intermediate

.. displayitem::
    :header: Track and visualize experiments
    :description: Learn how Fabric helps you remove boilerplate code for tracking metrics with a logger
    :button_link: logging.html
    :col_css: col-md-4
    :height: 170
    :tag: intermediate

.. displayitem::
    :header: Save and load model progress
    :description: Efficient saving and loading of model weights, training state, hyperparameters and more.
    :button_link: checkpoint.html
    :col_css: col-md-4
    :height: 170
    :tag: intermediate

.. displayitem::
    :header: Build your own Trainer
    :description: Take our Fabric Trainer template and customize it for your needs
    :button_link: https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/build_your_own_trainer
    :col_css: col-md-4
    :height: 170
    :tag: intermediate

.. raw:: html

        </div>
    </div>


***************
Advanced Topics
***************


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Use efficient gradient accumulation
    :description: Learn how to perform efficient gradient accumulation in distributed settings
    :button_link: ../advanced/gradient_accumulation.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Distribute communication
    :description: Learn all about communication primitives for distributed operation. Gather, reduce, broadcast, etc.
    :button_link: ../advanced/distributed_communication.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Use multiple models and optimizers
    :description: See how flexible Fabric is to work with multiple models and optimizers!
    :button_link: ../advanced/multiple_setup.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Speed up models by compiling them
    :description: Use torch.compile to speed up models on modern hardware
    :button_link: ../advanced/compile.html
    :col_css: col-md-4
    :height: 150
    :tag: advanced

.. displayitem::
    :header: Train models with billions of parameters
    :description: Train the largest models with FSDP/TP across multiple GPUs and machines
    :button_link: ../advanced/model_parallel/index.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Initialize models efficiently
    :description: Reduce the time and peak memory usage for model initialization
    :button_link: ../advanced/model_init.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Save and load very large models
    :description: Save and load very large models efficiently with distributed checkpoints
    :button_link: checkpoint/distributed_checkpoint.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. raw:: html

        </div>
    </div>
