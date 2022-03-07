.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.trainer.trainer import Trainer

.. _converting:

######################################
How to Organize PyTorch Into Lightning
######################################

To enable your code to work with Lightning, perform the following to organize PyTorch into Lightning.

--------

*******************************
1. Move your Computational Code
*******************************

Move the model architecture and forward pass to your :class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class LitModel(pl.LightningModule):
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

    class LitModel(pl.LightningModule):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

--------

*******************************
3. Configure the Training Logic
*******************************

Lightning automates the training loop for you and manages all of the associated components such as: epoch and batch tracking, optimizers and schedulers,
and metric reduction. As a user, you just need to define how your model behaves with a batch of training data within the
:meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` method. When using Lightning, simply override the
:meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` method which takes the current ``batch`` and the ``batch_idx``
as arguments. Optionally, it can take ``optimizer_idx`` if your LightningModule defines multiple optimizers within its
:meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers` hook.

.. testcode::

    class LitModel(pl.LightningModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

--------

*********************************
4. Configure the Validation Logic
*********************************

Lightning also automates the validation loop for you and manages all of the associated components such as: epoch and batch tracking, and metrics reduction. As a user,
you just need to define how your model behaves with a batch of validation data within the :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`
method. When using Lightning, simply override the :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` method which takes the current
``batch`` and the ``batch_idx`` as arguments. Optionally, it can take ``dataloader_idx`` if you configure multiple dataloaders.

To add an (optional) validation loop add logic to the
:meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` hook (make sure to use the hook parameters, ``batch`` and ``batch_idx`` in this case).

.. testcode::

    class LitModel(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", val_loss)

Additionally, you can run only the validation loop using :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate` method.

.. code-block:: python

    model = LitModel()
    trainer.validate(model)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for validation.

.. tip:: ``trainer.validate()`` loads the best checkpoint automatically by default if checkpointing was enabled during fitting.

--------

**************************
5. Configure Testing Logic
**************************

Lightning automates the testing loop for you and manages all the associated components, such as epoch and batch tracking, metrics reduction. As a user,
you just need to define how your model behaves with a batch of testing data within the :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`
method. When using Lightning, simply override the :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step` method which takes the current
``batch`` and the ``batch_idx`` as arguments. Optionally, it can take ``dataloader_idx`` if you configure multiple dataloaders.

.. testcode::

    class LitModel(pl.LightningModule):
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            self.log("test_loss", test_loss)

The test loop isn't used within :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`, therefore, you would need to explicitly call :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`.

.. code-block:: python

    model = LitModel()
    trainer.test(model)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for testing.

.. tip:: ``trainer.test()`` loads the best checkpoint automatically by default if checkpointing is enabled.

--------

*****************************
6. Configure Prediction Logic
*****************************

Lightning automates the prediction loop for you and manages all of the associated components such as epoch and batch tracking. As a user,
you just need to define how your model behaves with a batch of data within the :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step`
method. When using Lightning, simply override the :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` method which takes the current
``batch`` and the ``batch_idx`` as arguments. Optionally, it can take ``dataloader_idx`` if you configure multiple dataloaders.
If you don't override ``predict_step`` hook, it by default calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward` method on the batch.

.. testcode::

    class LitModel(LightningModule):
        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred

The predict loop will not be used until you call :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

.. code-block:: python

    model = LitModel()
    trainer.predict(model)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for testing.

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

If you still need to access the current device, you can use ``self.device`` anywhere in ``LightningModule`` except ``__init__`` and ``setup`` methods.
You are initializing a :class:`~torch.Tensor` within ``LightningModule.__init__`` method and want it to be moved to the device automatically you must
:meth:`~torch.nn.Module.register_buffer` to register it as a parameter.

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            z = torch.randn(4, 5, device=self.device)
            ...

--------

********************
8. Use your own data
********************

To use your DataLoaders, you can override the respective dataloader hooks in the :class:`~pytorch_lightning.core.lightning.LightningModule`:

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

Alternatively, you can pass your dataloaders in one of the following ways:

* Pass in the dataloaders explictly inside ``trainer.fit/.validate/.test/.predict`` calls.
* Use a :ref:`LightningDataModule <datamodules>`.

Checkout :ref:`data` doc to understand data management within Lightning.
