:orphan:

.. _jupyter_notebooks:

##############################################
Interactive Notebooks (Jupyter, Colab, Kaggle)
##############################################

**Audience:** Users looking to train models in interactive notebooks (Jupyter, Colab, Kaggle, etc.).


----


**********************
Lightning in notebooks
**********************

You can use the Lightning Trainer in interactive notebooks just like in a regular Python script, including multi-GPU training!

.. code-block:: python

    import lightning as L

    # Works in Jupyter, Colab and Kaggle!
    trainer = L.Trainer(accelerator="auto", devices="auto")


You can find many notebook examples on our :doc:`tutorials page <../tutorials>` too!


----


.. _jupyter_notebook_example:

************
Full example
************

Paste the following code block into a notebook cell:

.. code-block:: python

    import lightning as L
    from torch import nn, optim, utils
    import torchvision

    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


    class LitAutoEncoder(L.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = nn.functional.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=1e-3)

        def prepare_data(self):
            torchvision.datasets.MNIST(".", download=True)

        def train_dataloader(self):
            dataset = torchvision.datasets.MNIST(".", transform=torchvision.transforms.ToTensor())
            return utils.data.DataLoader(dataset, batch_size=64)


    autoencoder = LitAutoEncoder(encoder, decoder)
    trainer = L.Trainer(max_epochs=2, devices="auto")
    trainer.fit(model=autoencoder)


----


*********************
Multi-GPU Limitations
*********************

The multi-GPU capabilities in Jupyter are enabled by launching processes using the 'fork' start method.
It is the only supported way of multi-processing in notebooks, but also brings some limitations that you should be aware of.

Avoid initializing CUDA before .fit()
=====================================

Don't run torch CUDA functions before calling ``trainer.fit()`` in any of the notebook cells beforehand, otherwise your code may hang or crash.

.. code-block:: python

    # BAD: Don't run CUDA-related code before `.fit()`
    x = torch.tensor(1).cuda()
    torch.cuda.empty_cache()
    torch.cuda.is_available()

    trainer = L.Trainer(accelerator="cuda", devices=2)
    trainer.fit(model)


Move data loading code inside the hooks
=======================================

If you define/load your data in the main process before calling ``trainer.fit()``, you may see a slowdown or crashes (segmentation fault, SIGSEV, etc.).

.. code-block:: python

    # BAD: Don't load data in the main process
    dataset = MyDataset("data/")
    train_dataloader = torch.utils.data.DataLoader(dataset)

    trainer = L.Trainer(accelerator="cuda", devices=2)
    trainer.fit(model, train_dataloader)

The best practice is to move your data loading code inside the ``*_dataloader()`` hooks in the :class:`~lightning.pytorch.core.LightningModule` or :class:`~lightning.pytorch.core.datamodule.LightningDataModule` as shown in the :ref:`example above <jupyter_notebook_example>`.
