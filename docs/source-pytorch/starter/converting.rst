.. _converting:

######################################
How to Organize PyTorch Into Lightning
######################################

To enable your code to work with Lightning, perform the following to organize PyTorch into Lightning.

--------

*******************************
1. Keep Your Computational Code
*******************************

Keep your regular nn.Module architecture

.. testcode::

    import lightning as L
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class LitModel(nn.Module):
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

***************************
2. Configure Training Logic
***************************
In the training_step of the LightningModule configure how your training routine behaves with a batch of training data:

.. testcode::

    class LitModel(L.LightningModule):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.encoder(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

.. note:: If you need to fully own the training loop for complicated legacy projects, check out :doc:`Own your loop <../model/own_your_loop>`.

----

****************************************
3. Move Optimizer(s) and LR Scheduler(s)
****************************************
Move your optimizers to the :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` hook.

.. testcode::

    class LitModel(L.LightningModule):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

--------

***************************************
4. Organize Validation Logic (optional)
***************************************
If you need a validation loop, configure how your validation routine behaves with a batch of validation data:

.. testcode::

    class LitModel(L.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.encoder(x)
            val_loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", val_loss)

.. tip:: ``trainer.validate()`` loads the best checkpoint automatically by default if checkpointing was enabled during fitting.

--------

************************************
5. Organize Testing Logic (optional)
************************************
If you need a test loop, configure how your testing routine behaves with a batch of test data:

.. testcode::

    class LitModel(L.LightningModule):
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.encoder(x)
            test_loss = F.cross_entropy(y_hat, y)
            self.log("test_loss", test_loss)

--------

****************************************
6. Configure Prediction Logic (optional)
****************************************
If you need a prediction loop, configure how your prediction routine behaves with a batch of test data:

.. testcode::

    class LitModel(L.LightningModule):
        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self.encoder(x)
            return pred

--------

******************************************
7. Remove any .cuda() or .to(device) Calls
******************************************

Your :doc:`LightningModule <../common/lightning_module>` can automatically run on any hardware!

If you have any explicit calls to ``.cuda()`` or ``.to(device)``, you can remove them since Lightning makes sure that the data coming from :class:`~torch.utils.data.DataLoader`
and all the :class:`~torch.nn.Module` instances initialized inside ``LightningModule.__init__`` are moved to the respective devices automatically.
If you still need to access the current device, you can use ``self.device`` anywhere in your ``LightningModule`` except in the ``__init__`` and ``setup`` methods.

.. testcode::

    class LitModel(L.LightningModule):
        def training_step(self, batch, batch_idx):
            z = torch.randn(4, 5, device=self.device)
            ...

Hint: If you are initializing a :class:`~torch.Tensor` within the ``LightningModule.__init__`` method and want it to be moved to the device automatically you should call
:meth:`~torch.nn.Module.register_buffer` to register it as a parameter.

.. testcode::

    class LitModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.zeros(num_features))

--------

********************
8. Use your own data
********************
Regular PyTorch DataLoaders work with Lightning. For more modular and scalable datasets, check out :doc:`LightningDataModule <../data/datamodule>`.

----

************
Good to know
************

Additionally, you can run only the validation loop using :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate` method.

.. code-block:: python

    model = LitModel()
    trainer.validate(model)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for validation.


The test loop isn't used within :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`, therefore, you would need to explicitly call :meth:`~lightning.pytorch.trainer.trainer.Trainer.test`.

.. code-block:: python

    model = LitModel()
    trainer.test(model)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for testing.

.. tip:: ``trainer.test()`` loads the best checkpoint automatically by default if checkpointing is enabled.


The predict loop will not be used until you call :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`.

.. code-block:: python

    model = LitModel()
    trainer.predict(model)

.. note:: ``model.eval()`` and ``torch.no_grad()`` are called automatically for testing.

.. tip:: ``trainer.predict()`` loads the best checkpoint automatically by default if checkpointing is enabled.
