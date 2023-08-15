
.. _fully-sharded-training:

**********************
Fully Sharded Training
**********************

PyTorch has it's own version of `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_ which is upstreamed from their `fairscale <https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html>`__ project.
It was introduced in their `v1.11.0 release <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`_ but it is recommended to use it with PyTorch v1.12 or more and that's what
Lightning supports.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

Auto Wrapping
=============

Model layers should be wrapped in FSDP in a nested way to save peak memory and enable communication and computation overlapping. The
simplest way to do it is auto wrapping, which can serve as a drop-in replacement for DDP without changing the rest of the code. You don't
have to ``wrap`` layers manually as in the case of manual wrapping.

.. note::
    For users of PyTorch < 2.0: While initializing the optimizers inside ``configure_optimizers`` hook, make sure to use ``self.trainer.model.parameters()``, else
    PyTorch will raise an error. This is required because when you use auto-wrap, the model layers are sharded and your
    ``lightning_module.parameters()`` will return a generator with no params.

.. code-block:: python

    model = BoringModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="fsdp", precision=16)
    trainer.fit(model)


You can customize the strategy configuration by adjusting the arguments of :class:`~lightning.pytorch.strategies.FSDPStrategy` and pass that to the ``strategy`` argument inside the ``Trainer``.

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import FSDPStrategy

    # equivalent to passing `"fsdp_cpu_offload"`
    fsdp = FSDPStrategy(cpu_offload=True)
    trainer = pl.Trainer(strategy=fsdp, accelerator="gpu", devices=4)

    # configure the wrapping condition
    fsdp = FSDPStrategy(auto_wrap_policy={MyTransformerBlock})
    trainer = pl.Trainer(strategy=fsdp, accelerator="gpu", devices=4)


Read more `here <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/#auto-wrapping>`__.


Manual Wrapping
===============

Manual wrapping can be useful to explore complex sharding strategies by applying ``wrap`` selectively to some parts of the model. To activate
parameter sharding with manual wrapping, you can wrap your model using the ``wrap`` function. Internally in Lightning, we enable a context manager around the ``configure_model`` hook to make sure the ``wrap`` parameters are passed correctly.

When not using Fully Sharded, these ``wrap`` calls are a no-op. This means once the changes have been made, there is no need to remove the changes for other strategies.

``wrap`` simply wraps the module with a Fully Sharded Parallel class with the correct parameters from the Lightning context manager.

Here's an example using that uses ``wrap`` to create your model:

.. code-block:: python

    import torch
    import torch.nn as nn
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
    from torch.distributed.fsdp.wrap import wrap


    class MyModel(pl.LightningModule):
        def configure_model(self):
            self.linear_layer = nn.Linear(32, 32)
            self.block = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))

            # modules are sharded across processes
            # as soon as they are wrapped with `wrap`.
            # During the forward/backward passes, weights get synced across processes
            # and de-allocated once computation is complete, saving memory.

            # Wraps the layer in a Fully Sharded Wrapper automatically
            linear_layer = wrap(self.linear_layer)

            for i, layer in enumerate(self.block):
                self.block[i] = wrap(layer)

            self.model = nn.Sequential(linear_layer, nn.ReLU(), self.block)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="fsdp", precision=16)
    trainer.fit(model)

In this case, Lightning will not re-wrap your model, so you don't need to set ``FSDPStrategy(auto_wrap_policy=...)``.

Check out `this tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ to learn more about it.

----


Activation Checkpointing
========================

Activation checkpointing reduces GPU memory usage by avoiding the storage of intermediate activation tensors in
selected layers. The tradeoff is that computation cost for the backpropagation increases, as the dropped activations
need to be recomputed.

Enable checkpointing on large layers (like Transformers) by providing a policy:

.. code-block:: python

    from lightning.pytorch.strategies import FSDPStrategy

    fsdp = FSDPStrategy(activation_checkpointing_policy={MyTransformerBlock})
    trainer = pl.Trainer(strategy=fsdp, accelerator="gpu", devices=4)


You could also configure activation checkpointing manually inside the ``configure_model`` hook:

.. code-block:: python

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


    class MyModel(pl.LightningModule):
        ...

        def configure_model(self):
            # Same code as in the "Manual wrapping" snippet above
            ...
            apply_activation_checkpointing(self.model)

In this case, Lightning will not re-configure activation checkpointing, so you don't need to set ``FSDPStrategy(activation_checkpointing=...)``.

