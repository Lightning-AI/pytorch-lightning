###########################################
LightningLite - Stepping Stone to Lightning
###########################################


.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/lite/lightning_lite.gif
    :alt: Animation showing how to convert a standard training loop to a Lightning loop



:class:`~pytorch_lightning.lite.LightningLite` enables pure PyTorch users to scale their existing code
on any kind of device while retaining full control over their own loops and optimization logic.

:class:`~pytorch_lightning.lite.LightningLite` is the right tool for you if you match one of the two following descriptions:

- As a PyTorch user, I want to quickly scale my existing code to multiple devices with minimal code changes.

- As a PyTorch user, I would like to convert my existing code to the Lightning API, but a one-step transition might be too complex. I am looking for a stepping stone to ensure reproducibility during the transition.

Supported integrations
======================

:class:`~pytorch_lightning.lite.LightningLite` supports single and multiple models / optimizers.

#. ``accelerator``:
    * CPU.
    * GPU.
    * TPU.
#. ``strategy``:
    * ``dp``: Data Parallel.
    * ``ddp`` or ``ddp_spawn``: Distributed Data Parallel.
    * ``ddp_shared`` or ``ddp_sharded_spawn``: Distributed Data Parallel with Zero 2.
    * ``deepspeed``: Distributed Data Parallel with Zero 2 / 3.
#. ``precision``: ``float16`` and ``bfloat16`` with ``AMP`` or ``float64``.
#. ``clusters``: ``TorchElastic``, ``SLURM``, ``Kubeflow``, ``LSF``.


Coming in the near future:

#. ``accelerator``: IPUs.
#. ``strategy``: ``Horovod``, ``FSDP``.

################
Learn by example
################

My existing PyTorch code
========================

In the code below, we have a `BoringModel` containing a single linear layer trained on some random data for 10 epochs.
The `run` function contains a custom training and validation loops.

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset


    class BoringModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            x = self.layer(x)
            return torch.nn.functional.mse_loss(x, torch.ones_like(x))


    def configure_optimizers(module: nn.Module):
        return torch.optim.SGD(module.parameters(), lr=0.001)


    class RandomDataset(Dataset):
        def __init__(self, length: int, size: int):
            self.len = length
            self.data = torch.randn(length, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len


    def train_dataloader():
        return DataLoader(RandomDataset(64, 32))


    def val_dataloader():
        return DataLoader(RandomDataset(64, 32))


    def run(num_epochs, model, train_dataloader, val_dataloader):
        optimizer = configure_optimizers(model)

        for epoch in range(num_epochs):
            train_losses = []
            val_losses = []

            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss)

            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    val_losses.append(model(batch))

            print(f"{epoch}/{num_epochs}| Train Epoch Loss: {torch.stack(train_losses).mean()}")
            print(f"{epoch}/{num_epochs}| Valid Epoch Loss: {torch.stack(val_losses).mean()}")


    model = BoringModel()
    run(10, model, train_dataloader(), val_dataloader())

Convert to LightningLite
========================

Here are 4 required steps to convert to :class:`~pytorch_lightning.lite.LightningLite`.

1. Subclass :class:`~pytorch_lightning.lite.LightningLite` and override its :meth:`~pytorch_lightning.lite.LightningLite.run` method.
2. Copy / paste the body of your existing `run` function.
3. Apply ``self.setup`` over each model and optimizers pair, ``self.setup_dataloaders`` on all your dataloaders and replace ``loss.backward()`` by ``self.backward(loss)``
4. Instantiate your ``Lite`` and call its :meth:`~pytorch_lightning.lite.LightningLite.run` method.

.. code-block:: python

    from pytorch_lightning.lite import LightningLite


    class Lite(LightningLite):
        def run(self, num_epochs, model, train_dataloader, val_dataloader):
            optimizer = configure_optimizers(model)

            ###################################################################################
            # You would need to call `self.setup` to wrap `model` and `optimizer`. If you     #
            # have multiple models (c.f GAN), call `setup` for each one of them and their     #
            # associated optimizers.                                                          #
            model, optimizer = self.setup(model=model, optimizers=optimizer)
            ###################################################################################

            ###################################################################################
            # You would need to call `self.setup_dataloaders` to prepare the dataloaders      #
            # in case you are running in a distributed setting.                               #
            train_dataloader = self.setup_dataloaders(train_dataloader)
            val_dataloader = self.setup_dataloaders(val_dataloader)
            ###################################################################################

            for epoch in range(num_epochs):
                train_losses = []
                val_losses = []

                model.train()
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    loss = model(batch)
                    train_losses.append(loss)
                    ###########################################################################
                    # By calling `self.backward` directly, `LightningLite` will automate      #
                    # precision and device scaling.                                           #
                    self.backward(loss)
                    ###########################################################################
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_losses.append(model(batch))

                ###############################################################################
                # By calling `self.all_gather` directly, tensors will be transferred          #
                # across processes and concatenated.                                          #
                train_epoch_loss = self.all_gather(train_losses).mean()
                val_epoch_loss = self.all_gather(val_losses).mean()
                ###############################################################################

                print(f"{epoch}/{num_epochs}| Train Epoch Loss: {train_epoch_loss}")
                print(f"{epoch}/{num_epochs}| Valid Epoch Loss: {val_epoch_loss}")


    seed_everything(42)
    lite_model = BoringModel()
    lite = Lite()
    lite.run(lite_model, train_dataloader(), val_dataloader())

That's all. You can now train on any kind of device and scale your training.

The :class:`~pytorch_lightning.lite.LightningLite` take care of device management, so you don't have to.

You can remove any device specific logic within your code.

Here is how to train on 8 gpus with `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision.

.. code-block:: python

    seed_everything(42)
    lite_model = BoringModel()
    lite = Lite(strategy="ddp", devices=8, accelerator="gpu", precision="bf16")
    lite.run(lite_model, train_dataloader(), val_dataloader())


Here is how to use `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with 8 gpus and precision 16


.. code-block:: python

    seed_everything(42)
    lite_model = BoringModel()
    lite = Lite(strategy="deepspeed", devices=8, accelerator="gpu", precision=16)
    lite.run(lite_model, train_dataloader(), val_dataloader())


Distributed Training Pitfalls
=============================

The :class:`~pytorch_lightning.lite.LightningLite` provides you only with the tool to scale your training,
but there are several major challenges ahead of you now:

#. Processes divergence: This happens when processes execute different section of the code due to different if/else condition, race condition on existing files, etc... resulting in hanging.
#. Cross processes reduction: Wrongly reported metrics or gradients due mis-reduction.
#. Large sharded models: Instantiation, materialization and state management of large models.
#. Rank 0 only actions: Logging, profiling, etc..
#. Checkpointing / Early stopping / Callbacks: Ability to easily customize your training behaviour and make it stateful.
#. Batch-level fault tolerance training: Ability to resume from a failure as if it never happened.

If you are facing one of those challenges, you are already meeting the limit of :class:`~pytorch_lightning.lite.LightningLite`
and we strongly encourage you to slowly convert to Lightning, so you never have to worry about those.

LightningLite to Lightning
==========================

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
            x = self.forward(batch)
            self.log("train_loss", x)
            return x

        def validation_step(self, batch, batch_idx):
            x = self.forward(batch)
            self.log("val_loss", x)
            return x

        def configure_optimizers(self):
            return configure_optimizers(self)


    class BoringDataModule(LightningDataModule):
        def train_dataloader(self):
            return train_dataloader()

        def val_dataloader(self):
            return val_dataloader()


    seed_everything(42)
    model = BoringModel()
    lightning_module = LiftModel(model)
    datamodule = BoringDataModule()
    trainer = Trainer(max_epochs=10)
    trainer.fit(lightning_module, datamodule)
