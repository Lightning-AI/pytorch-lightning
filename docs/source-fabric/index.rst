.. include:: links.rst

####################
Welcome to ⚡ Fabric
####################

Fabric is the fast and lightweight way to scale PyTorch models without boilerplate code.

- Easily switch from running on CPU to GPU (Apple Silicon, CUDA, ...), TPU, multi-GPU or even multi-node training
- State-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed) and mixed precision out of the box
- Handles all the boilerplate device logic for you
- Brings useful tools to help you build a trainer (callbacks, logging, checkpoints, ...)
- Designed with multi-billion parameter models in mind

|

.. code-block:: diff

      import torch
      import torch.nn as nn
      from torch.utils.data import DataLoader, Dataset

    + from lightning.fabric import Fabric

      class PyTorchModel(nn.Module):
          ...

      class PyTorchDataset(Dataset):
          ...

    + fabric = Fabric(accelerator="cuda", devices=8, strategy="ddp")
    + fabric.launch()

    - device = "cuda" if torch.cuda.is_available() else "cpu"
      model = PyTorchModel(...)
      optimizer = torch.optim.SGD(model.parameters())
    + model, optimizer = fabric.setup(model, optimizer)
      dataloader = DataLoader(PyTorchDataset(...), ...)
    + dataloader = fabric.setup_dataloaders(dataloader)
      model.train()

      for epoch in range(num_epochs):
          for batch in dataloader:
              input, target = batch
    -         input, target = input.to(device), target.to(device)
              optimizer.zero_grad()
              output = model(input)
              loss = loss_fn(output, target)
    -         loss.backward()
    +         fabric.backward(loss)
              optimizer.step()
              lr_scheduler.step()


----


***********
Why Fabric?
***********

|
|

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/PyTorch-to-Fabric-Spectrum-2.svg
   :alt: Fabric spans across a large spectrum - from raw PyTorch all the way to high-level PyTorch Lightning
   :width: 100%

|
|

Fabric differentiates itself from a fully-fledged trainer like Lightning's `Trainer`_ in these key aspects:

**Fast to implement**
There is no need to restructure your code: Just change a few lines in the PyTorch script and you'll be able to leverage Fabric features.

**Maximum Flexibility**
Write your own training and/or inference logic down to the individual optimizer calls.
You aren't forced to conform to a standardized epoch-based training loop like the one in Lightning `Trainer`_.
You can do flexible iteration based training, meta-learning, cross-validation and other types of optimization algorithms without digging into framework internals.
This also makes it super easy to adopt Fabric in existing PyTorch projects to speed-up and scale your models without the compromise on large refactors.
Just remember: With great power comes a great responsibility.

**Maximum Control**
The Lightning `Trainer`_ has many built-in features to make research simpler with less boilerplate, but debugging it requires some familiarity with the framework internals.
In Fabric, everything is opt-in. Think of it as a toolbox: You take out the tools (Fabric functions) you need and leave the other ones behind.
This makes it easier to develop and debug your PyTorch code as you gradually add more features to it.
Fabric provides important tools to remove undesired boilerplate code (distributed, hardware, checkpoints, logging, ...), but leaves the design and orchestration fully up to you.


----

************
Installation
************

Fabric ships directly with Lightning. Install it with

.. code-block:: bash

    pip install lightning

For alternative ways to install, read the :doc:`installation guide <fundamentals/installation>`.



.. raw:: html

    <div style="display:none">

.. toctree::
    :maxdepth: 1
    :name: start
    :caption: Home

    self
    Install <fundamentals/installation>


.. toctree::
    :maxdepth: 1
    :caption: Get started in steps

    Basic skills <levels/basic>
    Intermediate skills <levels/intermediate>
    Advanced skills <levels/advanced>


.. toctree::
    :maxdepth: 1
    :caption: Core API Reference

    Fabric Arguments <api/fabric_args>
    Fabric Methods <api/fabric_methods>


.. toctree::
    :maxdepth: 1
    :caption: Full API Reference

    Accelerators <api/accelerators>
    Collectives <api/collectives>
    Environments <api/environments>
    Fabric <api/fabric>
    IO <api/io>
    Loggers <api/loggers>
    Precision <api/precision>
    Strategies <api/strategies>


.. toctree::
    :maxdepth: 1
    :name: more
    :caption: More

    Examples <examples/index>
    Glossary <glossary/index>
    How-tos <guide/index>
    Style Guide <fundamentals/code_structure>


.. raw:: html

    </div>
