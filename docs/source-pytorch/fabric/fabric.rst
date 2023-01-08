#############
Fabric (Beta)
#############


:class:`~lightning_fabric.fabric.Fabric` library allows you to scale any PyTorch model with just a few lines of code!
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

    <div class="tutorials-callout-container">
        <div class="row">

.. customcalloutitem::
    :header: Getting Started
    :description: Learn how to add Fabric to your PyTorch code
    :button_link: funtamentals/convert.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Accelerators
    :description: Take advantage of your hardware with a switch of a flag
    :button_link: funtamentals/accelerators.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Distributed Operation
    :description: Launch a Python script on multiple devices and machines
    :button_link: funtamentals/launch.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Fabric in Notebooks
    :description: Launch a Python script on multiple devices and machines
    :button_link: funtamentals/notebooks.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Code Structure
    :description: Best practices for setting up your training script with Fabric
    :button_link: funtamentals/code_structure.html
    :col_css: col-md-4

.. raw:: html

        </div>
    </div>


**********************
Build Your Own Trainer
**********************

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. customcalloutitem::
    :header: The LightningModule
    :description: Organize your code in a LightningModule and use it with Fabric
    :button_link: guide/lightning_module.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Callbacks
    :description: Make use of the Callback system in Fabric
    :button_link: guide/callbacks.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Logging
    :description: Learn how Fabric helps you remove logging boilerplate
    :button_link: guide/logging.html
    :col_css: col-md-4

.. customcalloutitem::
    :header: Trainer Template
    :description: Take our Fabric Trainer template and customize it for your needs
    :button_link: guide/trainer_template.html
    :col_css: col-md-4

.. raw:: html

        </div>
    </div>

***************
Advanced Topics
***************


.. customcalloutitem::
   :description: Learn how to benchmark PyTorch Lightning.
   :header: Benchmarking
   :button_link: benchmarking/benchmarks.html



********
Examples
********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Image Classification
   :description: Train an image classifier on the MNIST dataset
   :button_link: ...
   :col_css: col-md-4
   :height: 150
   :tag: basic


.. raw:: html

        </div>
    </div>


***
API
***

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. customcalloutitem::
    :header: Fabric Arguments
    :description: All configuration options for the Fabric object
    :button_link: X
    :col_css: col-md-4

.. customcalloutitem::
    :header: Fabric Methods
    :description: Explore all methods that Fabric offers
    :button_link: X
    :col_css: col-md-4

.. customcalloutitem::
    :header: Utilities
    :description: Useful utility functions that make your life easier
    :button_link: X
    :col_css: col-md-4

.. customcalloutitem::
    :header: Full API Reference
    :button_link: api_reference.html
    :col_css: col-md-4

.. raw:: html

        </div>
    </div>


----------

You can also easily use distributed collectives if required.

.. code-block:: python

    fabric = Fabric()

    # Transfer and concatenate tensors across processes
    fabric.all_gather(...)

    # Transfer an object from one process to all the others
    fabric.broadcast(..., src=...)

    # The total number of processes running across all devices and nodes.
    fabric.world_size

    # The global index of the current process across all devices and nodes.
    fabric.global_rank

    # The index of the current process among the processes running on the local node.
    fabric.local_rank

    # The index of the current node.
    fabric.node_rank

    # Whether this global rank is rank zero.
    if fabric.is_global_zero:
        # do something on rank 0
        ...

    # Wait for all processes to enter this call.
    fabric.barrier()


The code stays agnostic, whether you are running on CPU, on two GPUS or on multiple machines with many GPUs.

If you require custom data or model device placement, you can deactivate :class:`~lightning_fabric.fabric.Fabric`'s automatic placement by doing ``fabric.setup_dataloaders(..., move_to_device=False)`` for the data and ``fabric.setup(..., move_to_device=False)`` for the model.
Furthermore, you can access the current device from ``fabric.device`` or rely on :meth:`~lightning_fabric.fabric.Fabric.to_device` utility to move an object to the current device.
