#############
Fabric (Beta)
#############

Fabric allows you to scale any PyTorch model with just a few lines of code!
With Fabric you can easily scale your model to run on distributed devices using the strategy of your choice, while keeping full control over the training loop and optimization logic.

With only a few changes to your code, Fabric allows you to:

- Automatic placement of models and data onto the device
- Automatic support for mixed precision (speedup and smaller memory footprint)
- Seamless switching between hardware (CPU, GPU, TPU)
- State-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed)
- Easy-to-use launch command for spawning processes (DDP, torchelastic, etc)
- Multi-node support (TorchElastic, SLURM, and more)
- You keep full control of your training loop


.. code-block:: diff

      import torch
      import torch.nn as nn
      from torch.utils.data import DataLoader, Dataset

    + from lightning.fabric import Fabric

      class MyModel(nn.Module):
          ...

      class MyDataset(Dataset):
          ...

    + fabric = Fabric(accelerator="cuda", devices=8, strategy="ddp")
    + fabric.launch()

    - device = "cuda" if torch.cuda.is_available() else "cpu
      model = MyModel(...)
      optimizer = torch.optim.SGD(model.parameters())
    + model, optimizer = fabric.setup(model, optimizer)
      dataloader = DataLoader(MyDataset(...), ...)
    + dataloader = fabric.setup_dataloaders(dataloader)
      model.train()

      for epoch in range(num_epochs):
          for batch in dataloader:
    -         batch.to(device)
              optimizer.zero_grad()
              loss = model(batch)
    -         loss.backward()
    +         fabric.backward(loss)
              optimizer.step()


.. note:: Fabric is currently in Beta. Its API is subject to change based on feedback.


----------


************
Fundamentals
************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Getting Started
    :description: Learn how to add Fabric to your PyTorch code
    :button_link: fundamentals/convert.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Accelerators
    :description: Take advantage of your hardware with a switch of a flag
    :button_link: fundamentals/accelerators.html
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. displayitem::
    :header: Code Structure
    :description: Best practices for setting up your training script with Fabric
    :button_link: fundamentals/code_structure.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Distributed Operation
    :description: Launch a Python script on multiple devices and machines
    :button_link: fundamentals/launch.html
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. displayitem::
    :header: Fabric in Notebooks
    :description: Launch on multiple devices from within a Jupyter notebook
    :button_link: fundamentals/notebooks.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. raw:: html

        </div>
    </div>


----------


**********************
Build Your Own Trainer
**********************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: The LightningModule
    :description: Organize your code in a LightningModule and use it with Fabric
    :button_link: guide/lightning_module.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Callbacks
    :description: Make use of the Callback system in Fabric
    :button_link: guide/callbacks.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Logging
    :description: Learn how Fabric helps you remove boilerplate code for tracking metrics with a logger
    :button_link: guide/logging.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Trainer Template
    :description: Take our Fabric Trainer template and customize it for your needs
    :button_link: guide/trainer_template.html
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. raw:: html

        </div>
    </div>


----------


***************
Advanced Topics
***************

Comnig soon.



----------


.. _Fabric Examples:

********
Examples
********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Image Classification
    :description: Train an image classifier on the MNIST dataset
    :button_link: https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/image_classifier/README.md
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: GAN
    :description: Train a GAN that generates realistic human faces
    :button_link: https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/dcgan/README.md
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. displayitem::
    :header: Reinforcement Learning
    :description: Coming soon
    :col_css: col-md-4
    :height: 150

.. displayitem::
    :header: Active Learning
    :description: Coming soon
    :col_css: col-md-4
    :height: 150

.. displayitem::
    :header: Meta Learning
    :description: Coming soon
    :col_css: col-md-4
    :height: 150


.. raw:: html

        </div>
    </div>



----------


***
API
***

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Fabric Arguments
    :description: All configuration options for the Fabric object
    :button_link: api/fabric_args.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Fabric Methods
    :description: Explore all methods that Fabric offers
    :button_link: api/fabric_methods.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Utilities
    :description: Explore utility functions that make your life easier
    :button_link: api/utilities.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Full API Reference
    :description: Reference of all public classes, methods and functions. Useful for developers.
    :button_link: api/api_reference.html
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. raw:: html

        </div>
    </div>
