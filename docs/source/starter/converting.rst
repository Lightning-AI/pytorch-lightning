.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.trainer.trainer import Trainer

.. _converting:

**************************************
How to organize PyTorch into Lightning
**************************************

To enable your code to work with Lightning, here's how to organize PyTorch into Lightning

--------

1. Move your computational code
===============================
Move the model architecture and forward pass to your :doc:`lightning module <../common/lightning_module>`.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(28 * 28, 128)
            self.layer_2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.layer_1(x)
            x = F.relu(x)
            x = self.layer_2(x)
            return x

--------

2. Move the optimizer(s) and schedulers
=======================================
Move your optimizers to the :func:`~pytorch_lightning.core.LightningModule.configure_optimizers` hook.

.. testcode::

    class LitModel(LightningModule):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

--------

3. Find the train loop "meat"
=============================
Lightning automates most of the training for you, the epoch and batch iterations, all you need to keep is the training step logic.
This should go into the :func:`~pytorch_lightning.core.LightningModule.training_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case):

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

--------

4. Find the val loop "meat"
===========================
To add an (optional) validation loop add logic to the
:func:`~pytorch_lightning.core.LightningModule.validation_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).

.. testcode::

    class LitModel(LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            return val_loss

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for validation

--------

5. Find the test loop "meat"
============================
To add an (optional) test loop add logic to the
:func:`~pytorch_lightning.core.LightningModule.test_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).

.. testcode::

    class LitModel(LightningModule):
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for testing.

The test loop will not be used until you call.

.. code-block::

    trainer.test()

.. tip:: ``.test()`` loads the best checkpoint automatically

--------

6. Remove any .cuda() or to.device() calls
==========================================
Your :doc:`lightning module <../common/lightning_module>` can automatically run on any hardware!
