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

----------

Step 1: Build LightningModule
-----------------------------
A lightningModule defines

- Train loop
- Val loop
- Test loop
- Model + system architecture
- Optimizer

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
            result = pl.TrainResult(minimize=loss, checkpoint_on=loss)
            result.log('train_loss', loss, prog_bar=True)
            return result

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0005)

----------

Step 2: Fit with a Trainer
--------------------------
The trainer calls each loop at the correct time as needed. It also ensures it all works
well across any accelerator.

.. code-block:: python

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

The code above gives you the following for free:

- Automatic checkpoints
- Automatic Tensorboard (or the logger of your choice)
- Automatic CPU/GPU/TPU training
- Automatic 16-bit precision

All of it 100% rigorously tested and benchmarked

--------------

Training loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Under the hood, lightning does (in high-level pseudocode):

.. code-block:: python

    # init model
    model = LitModel()

    # enable training
    torch.set_grad_enabled(True)
    model.train()

    # get data + optimizer
    train_dataloader = model.train_dataloader()
    optimizer = model.configure_optimizers()

    for epoch in epochs:
        for batch in train_dataloader:
            # forward (TRAINING_STEP)
            loss = model.training_step(batch)

            # backward
            loss.backward()

            # apply and clear grads
            optimizer.step()
            optimizer.zero_grad()

----------

Adding a Validation loop
------------------------
To add an (optional) validation loop add the following function

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

Validation loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

Lightning automatically:

- Enables gradients and sets model to train() in the train loop
- Disables gradients and sets model to eval() in val loop
- After val loop ends, enables gradients and sets model to train()

-------------

Adding a Test loop
------------------
You might also need an optional test loop

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

Test loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^
Under the hood, lightning does the following in (pseudocode):

.. code-block:: python

    torch.set_grad_enabled(False)
    model.eval()
    test_outs = []
    for test_batch in model.test_dataloader:
        test_out = model.test_step(val_batch)
        test_outs.append(test_out)

    model.test_epoch_end(test_outs)

---------------

Data
----
Lightning operates on standard PyTorch Dataloaders (of any flavor). Use dataloaders in 3 ways.

Data in fit
^^^^^^^^^^^
Pass the dataloaders into `trainer.fit()`

.. code-block:: python

    trainer.fit(model, train_dataloader, val_dataloader)

Data in LightningModule
^^^^^^^^^^^^^^^^^^^^^^^
For fast research prototyping, it might be easier to link the model with the dataloaders.

.. code-block:: python

    class LitModel(pl.LightningModule):

        def train_dataloader(self):
            # your train transforms
            return DataLoader(YOUR_DATASET)

        def val_dataloader(self):
            # your val transforms
            return DataLoader(YOUR_DATASET)

        def test_dataloader(self):
            # your test transforms
            return DataLoader(YOUR_DATASET)

And fit like so:

.. code-block:: python

    model = LitModel()
    trainer.fit(model)

DataModule
^^^^^^^^^^
A more reusable approach is to define a DataModule which is simply a collection of all 3 data splits but
also captures:

- download instructions.
- processing.
- splitting.
- etc...

.. code-block:: python

    class MyDataModule(pl.DataModule):

        def __init__(self):
            ...

        def train_dataloader(self):
            # your train transforms
            return DataLoader(YOUR_DATASET)

        def val_dataloader(self):
            # your val transforms
            return DataLoader(YOUR_DATASET)

        def test_dataloader(self):
            # your test transforms
            return DataLoader(YOUR_DATASET)

And train like so:

.. code-block:: python

    dm = MyDataModule()
    trainer.fit(model, dm)

When doing distributed training, Datamodules have two optional arguments for granular control
over download/prepare/splitting data

.. code-block:: python

    class MyDataModule(pl.DataModule):

        def prepare_data(self):
            # called only on 1 GPU
            download()
            tokenize()
            etc()

         def setup(self):
            # called on every GPU (assigning state is OK)
            self.train = ...
            self.val = ...

         def train_dataloader(self):
            # do more...
            return self.train

Building models based on Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Datamodules are the recommended approach when building models based on the data.

First, define the information that you might need.

.. code-block:: python

    class MyDataModule(pl.DataModule):

        def __init__(self):
            super().__init__()
            self.train_dims = None
            self.vocab_size = 0

        def prepare_data(self):
            download_dataset()
            tokenize()
            build_vocab()

        def setup(self):
            vocab = load_vocab
            self.vocab_size = len(vocab)

            self.train, self.val, self.test = load_datasets()
            self.train_dims = self.train.next_batch.size()

        def train_dataloader(self):
            transforms = ...
            return DataLoader(self.train, transforms)

        def val_dataloader(self):
            transforms = ...
            return DataLoader(self.val, transforms)

        def test_dataloader(self):
            transforms = ...
            return DataLoader(self.test, transforms)

Next, materialize the data and build your model

.. code-block:: python

    # build module
    dm = MyDataModule()
    dm.prepare_data()
    dm.setup()

    # pass in the properties you want
    model = LitModel(image_width=dm.train_dims[0], image_height=dm.train_dims[1], vocab_length=dm.vocab_size)

    # train
    trainer.fit(model, dm)

-----------------

Logging/progress bar
--------------------
Lightning has built-in logging to any of the supported loggers or progress bar.

Log in train loop
^^^^^^^^^^^^^^^^^
To log from the training loop use the `TrainResult` object

.. code-block:: python

    def training_step(self, batch, batch_idx):
        loss = ...
        acc = ...

        # pick what to minimize
        result = pl.TrainResult(minimize=loss)

        # logs metric at the end of every training step (batch) to the tensorboard or user-specified logger
        result.log('train_loss', loss)

        # log to the progress bar only
        result.log('train_acc', acc, prog_bar=True, logger=False)

Then boot up your logger or tensorboard instance to view training logs

.. code-block:: bash

    tensorboard --logdir ./lightning_logs

.. warning:: Refreshing the progress bar too frequently in Jupyter notebooks or Colab may freeze your UI.

.. note:: TrainResult defaults to logging on every step, set `on_epoch` to also log the metric for the full epoch

Log in Val/Test loop
^^^^^^^^^^^^^^^^^^^^
To log from the validation or test loop use a similar approach

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        loss = ...
        acc = ...

        # pick what to minimize
        result = pl.EvalResult(checkpoint_on=acc, early_stop_on=loss)

        # log the val loss averaged across the full epoch
        result.log('val_loss', loss)

        # log the val acc at each step AND for the full epoch (mean)
        result.log('val_acc', acc, prog_bar=True, logger=True, on_epoch=True, on_step=True)

.. note:: EvalResult defaults to logging for the full epoch, use `reduce_fx=torch.mean` to specify a different function.

-----------------

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

---------------

Summary
-------
In short, by refactoring your PyTorch code:

1.  You STILL keep pure PyTorch.
2.  You DON't lose any flexibility.
3.  You can get rid of all of your boilerplate.
4.  You make your code generalizable to any hardware.
5.  Your code is now readable and easier to reproduce (ie: you help with the reproducibility crisis).
6.  Your LightningModule is still just a pure PyTorch module.
