:orphan:

#################
Convert to Fabric
#################

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

|

That's it! You can now train on any device at any scale with a switch of a flag.
Check out our examples that use Fabric:

- `Image Classification <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/image_classifier/README.md>`_
- `Generative Adversarial Network (GAN) <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/dcgan/README.md>`_


Here is how you run DDP with 8 GPUs and `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: bash

    lightning run model ./path/to/train.py --strategy=ddp --devices=8 --accelerator=cuda --precision="bf16"

Or `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with mixed precision:

.. code-block:: bash

     lightning run model ./path/to/train.py --strategy=deepspeed --devices=8 --accelerator=cuda --precision=16

:class:`~lightning_fabric.fabric.Fabric` can also figure it out automatically for you!

.. code-block:: bash

    lightning run model ./path/to/train.py --devices=auto --accelerator=auto --precision=16
