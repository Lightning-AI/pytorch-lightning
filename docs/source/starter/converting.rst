.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.trainer.trainer import Trainer

.. _converting:


######################################
How to organize PyTorch into Lightning
######################################

To enable your code to work with Lightning, here's how to organize PyTorch into Lightning:

--------

*******************************
1. Move your Computational code
*******************************

Move the model architecture and forward pass to your :class:`~pytorch_lightning.core.lightning.LightningModule`.

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

********************************************
2. Move the Optimizer(s) and LR Scheduler(s)
********************************************

Move your optimizers to the :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers` hook.

.. testcode::

    class LitModel(LightningModule):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

--------

*****************************
3. Find the Train Loop "meat"
*****************************

Lightning automates most of the training for you, the epoch and batch iterations, all you need to keep is the ``training_step`` logic.
This should go into the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

--------

***************************
4. Find the Val Loop "meat"
***************************

To add an (optional) validation loop add logic to the
:meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).

.. testcode::

    class LitModel(LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", val_loss)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for validation.

--------

****************************
5. Find the Test Loop "meat"
****************************

To add an (optional) test loop add logic to the
:meth:`~pytorch_lightning.core.lightning.LightningModule.test_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).

.. testcode::

    class LitModel(LightningModule):
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            self.log("test_loss", test_loss)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for testing.

The test loop will not be used until you call.

.. code-block:: python

    trainer.test()

.. tip:: ``trainer.test()`` loads the best checkpoint automatically by default if checkpointing is enabled.

--------

*******************************
6. Find the Predict Loop "meat"
*******************************

To add an (optional) prediction loop add logic to the
:meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).
If you don't override ``predict_step`` hook, it by default calls ``forward`` method on the batch.

.. testcode::

    class LitModel(LightningModule):
        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for prediction.

The predict loop will not be used until you call.

.. code-block:: python

    trainer.predict()

.. tip:: ``trainer.predict()`` loads the best checkpoint automatically by default if checkpointing is enabled.

--------

******************************************
7. Remove any .cuda() or .to(device) Calls
******************************************

Your :doc:`LightningModule <../common/lightning_module>` can automatically run on any hardware!

If you have any explicit calls to ``.cuda()`` or ``.to(device)``, you can remove them since Lightning makes sure that the data coming from :class:`~torch.utils.data.DataLoader`
and all the :class:`~torch.nn.Module` instances initialized inside ``LightningModule.__init__`` are moved to the respective devices automatically.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.zeros(num_features))

If you still need to access the current device, you can use ``self.device`` anywhere in ``LightningModule`` except ``__init__`` method. You are initializing a
:class:`~torch.Tensor` within ``LightningModule.__init__`` method and want it to be moved to the device automatically you must :meth:`~torch.nn.Module.register_buffer`
to register it as a parameter.

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            z = torch.randn(4, 5, device=self.device)
            ...

--------

**************
8. Plugin Data
**************

To plugin your DataLoaders, you can override the respective dataloader hooks:

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):
            return DataLoader(...)

        def val_dataloader(self):
            return DataLoader(...)

        def test_dataloader(self):
            return DataLoader(...)

        def predict_dataloader(self):
            return DataLoader(...)

Additionally, you can also plugin your dataloaders using one of the following ways:

* Pass in the dataloaders explictly inside ``trainer.fit/.validate/.test/.predict`` calls.
* Use :ref:`LightningDataModule <datamodules>`.

Checkout :ref:`data` doc to understand data management within Lightning.
