##############################
Convert PyTorch code to Fabric
##############################

Here are five easy steps to let :class:`~lightning.fabric.fabric.Fabric` scale your PyTorch models.

**Step 1:** Create the :class:`~lightning.fabric.fabric.Fabric` object at the beginning of your training code.

.. code-block:: python

    from lightning.fabric import Fabric

    fabric = Fabric()

**Step 2:** Call :meth:`~lightning.fabric.fabric.Fabric.launch` if you intend to use multiple devices (e.g., multi-GPU).

.. code-block:: python

    fabric.launch()

**Step 3:** Call :meth:`~lightning.fabric.fabric.Fabric.setup` on each model and optimizer pair and :meth:`~lightning.fabric.fabric.Fabric.setup_dataloaders` on all your data loaders.

.. code-block:: python

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

**Step 4:** Remove all ``.to`` and ``.cuda`` calls since :class:`~lightning.fabric.fabric.Fabric` will take care of it.

.. code-block:: diff

  - model.to(device)
  - batch.to(device)

**Step 5:** Replace ``loss.backward()`` by ``fabric.backward(loss)``.

.. code-block:: diff

  - loss.backward()
  + fabric.backward(loss)


These are all code changes required to prepare your script for Fabric.
You can now simply run from the terminal:

.. code-block:: bash

    python path/to/your/script.py

|

All steps combined, this is how your code will change:

.. code-block:: diff

      import torch
      from lightning.pytorch.demos import WikiText2, Transformer
    + import lightning as L

    - device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    + fabric = L.Fabric(accelerator="cuda", devices=8, strategy="ddp")
    + fabric.launch()

      dataset = WikiText2()
      dataloader = torch.utils.data.DataLoader(dataset)
      model = Transformer(vocab_size=dataset.vocab_size)
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    - model = model.to(device)
    + model, optimizer = fabric.setup(model, optimizer)
    + dataloader = fabric.setup_dataloaders(dataloader)

      model.train()
      for epoch in range(20):
          for batch in dataloader:
              input, target = batch
    -         input, target = input.to(device), target.to(device)
              optimizer.zero_grad()
              output = model(input, target)
              loss = torch.nn.functional.nll_loss(output, target.view(-1))
    -         loss.backward()
    +         fabric.backward(loss)
              optimizer.step()


That's it! You can now train on any device at any scale with a switch of a flag.
Check out our before-and-after example for `image classification <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/image_classifier/README.md>`_ and many more :doc:`examples <../examples/index>` that use Fabric.


----


****************
Optional changes
****************

Here are a few optional upgrades you can make to your code, if applicable:

- Replace ``torch.save()`` and ``torch.load()`` with Fabric's :doc:`save and load methods <../guide/checkpoint/checkpoint>`.
- Replace collective operations from ``torch.distributed`` (barrier, broadcast, etc.) with Fabric's :doc:`collective methods <../advanced/distributed_communication>`.
- Use Fabric's :doc:`no_backward_sync() context manager <../advanced/gradient_accumulation>` if you implemented gradient accumulation.
- Initialize your model under the :doc:`init_module() <../advanced/model_init>` context manager.


----


**********
Next steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Examples
    :description: See examples across computer vision, NLP, RL, etc.
    :col_css: col-md-4
    :button_link: ../examples/index.html
    :height: 150
    :tag: basic

.. displayitem::
    :header: Accelerators
    :description: Take advantage of your hardware with a switch of a flag
    :button_link: accelerators.html
    :col_css: col-md-4
    :height: 150
    :tag: basic

.. displayitem::
    :header: Build your own Trainer
    :description: Learn how to build a trainer tailored for you
    :col_css: col-md-4
    :button_link: ../levels/intermediate
    :height: 150
    :tag: intermediate

.. raw:: html

        </div>
    </div>
