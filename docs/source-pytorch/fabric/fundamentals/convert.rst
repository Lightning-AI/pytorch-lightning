:orphan:

##############################
Convert PyTorch code to Fabric
##############################

Here are five easy steps to let :class:`~lightning_fabric.fabric.Fabric` scale your PyTorch models.

**Step 1:** Create the :class:`~lightning_fabric.fabric.Fabric` object at the beginning of your training code.

.. code-block:: python

    from lightning.fabric import Fabric

    fabric = Fabric()

**Step 2:** Call :meth:`~lightning_fabric.fabric.Fabric.setup` on each model and optimizer pair and :meth:`~lightning_fabric.fabric.Fabric.setup_dataloaders` on all your dataloaders.

.. code-block:: python

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

**Step 3:** Remove all ``.to`` and ``.cuda`` calls since :class:`~lightning_fabric.fabric.Fabric` will take care of it.

.. code-block:: diff

  - model.to(device)
  - batch.to(device)

**Step 4:** Replace ``loss.backward()`` by ``fabric.backward(loss)``.

.. code-block:: diff

  - loss.backward()
  + fabric.backward(loss)

**Step 5:** Run the script from the terminal with

.. code-block:: bash

    lightning run model path/to/train.py``

or use the :meth:`~lightning_fabric.fabric.Fabric.launch` method in a notebook.
Learn more about :doc:`launching distributed training <launch>`.

|

All steps combined, this is how your code will change:

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


That's it! You can now train on any device at any scale with a switch of a flag.
Check out our before-and-after example for `image classification <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/image_classifier/README.md>`_ and many more :ref:`examples <Fabric Examples>` that use Fabric.
