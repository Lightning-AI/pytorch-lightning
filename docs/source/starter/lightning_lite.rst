###########################################
LightningLite - Stepping Stone to Lightning
###########################################


.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/lite/lightning_lite.gif
    :alt: Animation showing how to convert a standard training loop to a Lightning loop
    :width: 600px
    :align: center

|

:class:`~pytorch_lightning.lite.LightningLite` enables pure PyTorch users to scale their existing code
on any kind of device while retaining full control over their own loops and optimization logic.

:class:`~pytorch_lightning.lite.LightningLite` is the right tool for you if you match one of the two following descriptions:

- I want to quickly scale my existing code to multiple devices with minimal code changes.
- I would like to convert my existing code to the Lightning API, but a full path to Lightning transition might be too complex. I am looking for a stepping stone to ensure reproducibility during the transition.

**********************
Supported Integrations
**********************

:class:`~pytorch_lightning.lite.LightningLite` supports single and multiple models and optimizers.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - LightningLite arguments
     - Possible choices
   * - ``accelerator``
     - ``cpu``, ``gpu``, ``tpu``, ``auto``
   * - ``strategy``
     - ``dp``, ``ddp``, ``ddp_spawn``, ``ddp_sharded``, ``ddp_sharded_spawn``, ``deepspeed``
   * - ``precision``
     - ``16``, ``bf16``, ``32``, ``64``
   * - ``clusters``
     - ``TorchElastic``, ``SLURM``, ``Kubeflow``, ``LSF``


Coming soon: IPU accelerator, support for Horovod as a strategy and fully sharded training.


****************
Learn by example
****************




My existing PyTorch code
========================

The ``run`` function contains custom training loop used to train ``MyModel`` on ``MyDataset`` for ``num_epochs`` epochs.

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    def run(num_epochs: int):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MyModel(...).to(device)
        optimizer = torch.optim.SGD(model.parameters(), ...)

        dataloader = DataLoader(MyDataset(...), ...)

        model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()


    run(10)

Convert to LightningLite
========================

Here are 4 required steps to convert to :class:`~pytorch_lightning.lite.LightningLite`.

1. Subclass :class:`~pytorch_lightning.lite.LightningLite` and override its :meth:`~pytorch_lightning.lite.LightningLite.run` method.
2. Move the body of your existing `run` function.
3. Apply :meth:`~pytorch_lightning.lite.LightningLite.setup` over each model and optimizers pair, :meth:`~pytorch_lightning.lite.LightningLite.setup_dataloaders` on all your dataloaders and replace ``loss.backward()`` by ``self.backward(loss)``
4. Instantiate your :class:`~pytorch_lightning.lite.LightningLite` and call its :meth:`~pytorch_lightning.lite.LightningLite.run` method.


.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning.lite import LightningLite


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    class Lite(LightningLite):
        def run(self, num_epochs: int):

            model = MyModel(...)
            optimizer = torch.optim.SGD(model.parameters(), ...)

            model, optimizer = self.setup(model, optimizer)

            dataloader = DataLoader(MyDataset(...), ...)
            dataloader = self.setup_dataloaders(dataloader)

            model.train()
            for epoch in range(num_epochs):
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss = model(batch)
                    self.backward(loss)
                    optimizer.step()


    Lite(...).run(10)


That's all. You can now train on any kind of device and scale your training.
The :class:`~pytorch_lightning.lite.LightningLite` takes care of device management, so you don't have to.
You can remove any device specific logic within your code.
Here is how to train on 8 GPUs with `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: python

    Lite(strategy="ddp", devices=8, accelerator="gpu", precision="bf16").run(10)

Here is how to use `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with 8 GPUs and precision 16:

.. code-block:: python

    Lite(strategy="deepspeed", devices=8, accelerator="gpu", precision=16).run(10)

Lightning can also figure it automatically for you!

.. code-block:: python

    Lite(devices="auto", accelerator="auto", precision=16).run(10)


You can also easily use distributed collectives if required.
Here is an example while running on 256 GPUs.

.. code-block:: python

    class Lite(LightningLite):
        def run(self):

            # Transfer and concatenate tensors across processes
            self.all_gather(...)

            # Transfer an object from one process to all the others
            self.broadcast(..., src=...)

            # The total number of processes running across all devices and nodes.
            self.world_size

            # The global index of the current process across all devices and nodes.
            self.global_rank

            # The index of the current process among the processes running on the local node.
            self.local_rank

            # The index of the current node.
            self.node_rank

            # Wether this global rank is rank zero.
            if self.is_global_zero:
                # do something on rank 0
                ...

            # Wait for all processes to enter this call.
            self.barrier()

            # Reduce a boolean decision across processes.
            self.reduce_decision(...)


    Lite(strategy="ddp", gpus=8, num_nodes=32, accelerator="gpu").run()


.. note:: We recommend instantiating the models within the :meth:`~pytorch_lightning.lite.LightningLite.run` method as large models would cause OOM Error otherwise.


Distributed Training Pitfalls
=============================

The :class:`~pytorch_lightning.lite.LightningLite` provides you only with the tool to scale your training,
but there are several major challenges ahead of you now:


.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - Processes divergence
     - This happens when processes execute different section of the code due to different if/else condition, race condition on existing files, etc., resulting in hanging.
   * - Cross processes reduction
     - Wrongly reported metrics or gradients due mis-reduction.
   * - Large sharded models
     - Instantiation, materialization and state management of large models.
   * - Rank 0 only actions
     - Logging, profiling, etc.
   * - Checkpointing / Early stopping / Callbacks
     - Ability to easily customize your training behaviour and make it stateful.
   * - Batch-level fault tolerance training
     - Ability to resume from a failure as if it never happened.


If you are facing one of those challenges then you are already meeting the limit of :class:`~pytorch_lightning.lite.LightningLite`.
We recommend you to convert to :doc:`Lightning <../starter/new-project>`, so you never have to worry about those.

Convert to Lightning
====================

The :class:`~pytorch_lightning.lite.LightningLite` is a stepping stone to transition fully to the Lightning API and benefits
from its hundreds of features.

.. code-block:: python

    from pytorch_lightning import LightningDataModule, LightningModule, Trainer


    class LiftModel(LightningModule):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, x):
            return self.module(x)

        def training_step(self, batch, batch_idx):
            loss = self(batch)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            loss = self(batch)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.001)


    class BoringDataModule(LightningDataModule):
        def __init__(self, dataset: Dataset):
            super().__init__()
            self.dataset = dataset

        def train_dataloader(self):
            return DataLoader(self.dataset)


    seed_everything(42)
    model = MyModel(...)
    lightning_module = LiftModel(model)
    dataset = MyDataset(...)
    datamodule = BoringDataModule(dataset)
    trainer = Trainer(max_epochs=10)
    trainer.fit(lightning_module, datamodule=datamodule)
