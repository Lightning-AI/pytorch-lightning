.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms


Quick Start
===========

PyTorch Lightning is nothing more than organized PyTorch code.
Once you've organized it into a LightningModule, it automates most of the training for you.

To illustrate, here's the typical PyTorch project structure organized in a LightningModule.

.. figure:: /_images/mnist_imgs/pt_to_pl.jpg
   :alt: Convert from PyTorch to Lightning


Step 1: Define a LightningModule
---------------------------------

.. testcode::
    :skipif: not TORCHVISION_AVAILABLE


    import pytorch_lightning as pl
    from pytorch_lightning.metrics.functional import accuracy

    class LitModel(pl.LightningModule):

        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = pl.TrainResult(minimize=loss)
            result.log('train_loss', loss, prog_bar=True)
            return result

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0005)


Step 2: Fit with a Trainer
--------------------------

.. testcode::
    :skipif: torch.cuda.device_count() < 8

    # dataloader
    train_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), shuffle=True)

    # init model
    model = LitModel()

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(gpus=8, num_nodes=1)
    trainer.fit(
        model,
        train_loader,
    )

    # to use advanced features such as GPUs/TPUs/16 bit you have to change NO CODE
    trainer = pl.Trainer(tpu_cores=8, precision=16)

What happens under the hood
---------------------------
Under the hood, lightning does (in high-level pseudocode):

.. code-block:: python

    model = LitModel()
    torch.set_grad_enabled(True)
    model.train()
    train_dataloader = model.train_dataloader()
    optimizer = model.configure_optimizers()

    for epoch in epochs:
        train_outs = []
        for batch in train_dataloader:
            loss = model.training_step(batch)
            loss.backward()
            train_outs.append(loss.detach())

            optimizer.step()
            optimizer.zero_grad()

        # TrainResult automatically gathers metrics to log for the epoch

Adding a Validation loop
------------------------
To also add a validation loop add the following functions

.. testcode::

    class LitModel(LightningModule):

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
            result.log('val_ce', loss)
            result.log('val_acc', accuracy(y_hat, y))
            return result

And now the trainer will call the validation loop automatically

.. code-block:: python

    # pass in the val dataloader to the trainer as well
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader
    )

Under the hood in pseudocode, lightning does the following:

.. code-block:: python

    # ...
    for batch in train_dataloader:
        loss = model.training_step()
        loss.backward()
        # ...

        if validate_at_some_point:
            torch.set_grad_enabled(False)
            model.eval()
            val_outs = []
            for val_batch in model.val_dataloader:
                val_out = model.validation_step(val_batch)
                val_outs.append(val_out)

            model.validation_epoch_end(val_outs)
            torch.set_grad_enabled(True)
            model.train()

The beauty of Lightning is that it handles the details of when to validate, when to call .eval(),
turning off gradients, detaching graphs, making sure you don't enable shuffle for val, etc...

.. note:: Lightning removes all the million details you need to remember during research

Test loop
---------
You might also need a test loop

.. testcode::

    class LitModel(LightningModule):

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
            result.log('test_ce', loss)
            result.log('test_acc', accuracy(y_hat, y), prog_bar=True)
            return result

However, this time you need to specifically call test (this is done so you don't use the test set by mistake)

.. code-block:: python

    # OPTION 1:
    # test after fit
    trainer.fit(model)
    trainer.test(test_dataloaders=test_dataloader)

    # OPTION 2:
    # test after loading weights
    model = LitModel.load_from_checkpoint(PATH)
    trainer = Trainer(tpu_cores=1)
    trainer.test(test_dataloaders=test_dataloader)

Again, under the hood, lightning does the following in (pseudocode):

.. code-block:: python

    torch.set_grad_enabled(False)
    model.eval()
    test_outs = []
    for test_batch in model.test_dataloader:
        test_out = model.test_step(val_batch)
        test_outs.append(test_out)

    model.test_epoch_end(test_outs)


Why do you need Lightning?
--------------------------
Notice the code above has nothing about .cuda() or 16-bit or early stopping or logging, etc...
This is where Lightning adds a ton of value.

Without changing a SINGLE line of your code, you can now do the following with the above code

.. code-block:: python

    # train on TPUs using 16 bit precision with early stopping
    # using only half the training data and checking validation every quarter of a training epoch
    trainer = Trainer(
        tpu_cores=8,
        precision=16,
        early_stop_checkpoint=True,
        limit_train_batches=0.5,
        val_check_interval=0.25
    )

    # train on 256 GPUs
    trainer = Trainer(
        gpus=8,
        num_nodes=32
    )

    # train on 1024 CPUs across 128 machines
    trainer = Trainer(
        num_processes=8,
        num_nodes=128
    )

And the best part is that your code is STILL just PyTorch... meaning you can do anything you
would normally do.

.. code-block:: python

    model = LitModel()
    model.eval()

    y_hat = model(x)

    model.anything_you_can_do_with_pytorch()

Summary
-------
In short, by refactoring your PyTorch code:

1.  You STILL keep pure PyTorch.
2.  You DON't lose any flexibility.
3.  You can get rid of all of your boilerplate.
4.  You make your code generalizable to any hardware.
5.  Your code is now readable and easier to reproduce (ie: you help with the reproducibility crisis).
6.  Your LightningModule is still just a pure PyTorch module.
