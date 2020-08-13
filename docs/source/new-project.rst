.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader


Quick Start
===========

PyTorch Lightning is nothing more than organized PyTorch code.

Once you've organized it into a LightningModule, it automates most of the training for you.

To illustrate, here's the typical PyTorch project structure organized in a LightningModule.

.. raw:: html

    <video width="800" controls src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_module_vid.m4v"></video>

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
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0005)

----------

Step 2: Fit with a Trainer
--------------------------
The trainer calls each loop at the correct time as needed. It also ensures it all works
well across any accelerator.

.. raw:: html

    <video width="800" controls src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_trainer_mov.m4v"></video>

|

Here's an example of using the Trainer:

.. code-block:: python

    # dataloader
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

    # init model
    model = LitModel()

    # most basic trainer, uses good defaults
    trainer = pl.Trainer()
    trainer.fit(model, train_loader)

Using GPUs/TPUs
^^^^^^^^^^^^^^^
It's trivial to use GPUs or TPUs in Lightning. There's NO NEED to change your code, simply change the Trainer options.

.. code-block:: python

    # train on 1, 2, 4, n GPUs
    Trainer(gpus=1)
    Trainer(gpus=2)
    Trainer(gpus=8, num_nodes=n)

    # train on TPUs
    Trainer(tpu_cores=8)
    Trainer(tpu_cores=128)

    # even half precision
    Trainer(gpus=2, precision=16)

The code above gives you the following for free:

- Automatic checkpoints
- Automatic Tensorboard (or the logger of your choice)
- Automatic CPU/GPU/TPU training
- Automatic 16-bit precision

All of it 100% rigorously tested and benchmarked

--------------

Lightning under the hood
^^^^^^^^^^^^^^^^^^^^^^^^
Lightning is designed for state of the art research ideas by researchers and research engineers from top labs.

A LightningModule handles advances cases by allowing you to override any critical part of training
via hooks that are called on your LightningModule.

.. raw:: html

    <video width="800" controls src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_callbacks_mov.m4v"></video>

----------------

Training loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is the training loop pseudocode that lightning does under the hood:

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

Main take-aways:

- Lightning sets .train() and enables gradients when entering the training loop.
- Lightning iterates over the epochs automatically.
- Lightning iterates the dataloaders automatically.
- Training_step gives you full control of the main loop.
- .backward(), .step(), .zero_grad() are called for you. BUT, you can override this if you need manual control.

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

            result = pl.EvalResult(checkpoint_on=loss)
            result.log('val_loss', loss)
            return result

.. note:: EvalResult is a plain Dict, with convenience functions for logging

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
            # disable grads + batchnorm + dropout
            torch.set_grad_enabled(False)
            model.eval()

            val_outs = []
            for val_batch in model.val_dataloader:
                val_out = model.validation_step(val_batch)
                val_outs.append(val_out)
            model.validation_epoch_end(val_outs)

            # enable grads + batchnorm + dropout
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

            result = pl.EvalResult()
            result.log('test_loss', loss)
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
    trainer = Trainer()
    trainer.test(test_dataloaders=test_dataloader)

Test loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^
Under the hood, lightning does the following in (pseudocode):

.. code-block:: python

    # disable grads + batchnorm + dropout
    torch.set_grad_enabled(False)
    model.eval()

    test_outs = []
    for test_batch in model.test_dataloader:
        test_out = model.test_step(val_batch)
        test_outs.append(test_out)

    model.test_epoch_end(test_outs)

    # enable grads + batchnorm + dropout
    torch.set_grad_enabled(True)
    model.train()

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

Here's an illustration that explains how to refactor your code into reusable DataModules.

.. raw:: html

    <video width="800" controls src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_dm_vid.m4v"></video>

|

And the matching code:

|

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
    model = LitModel(image_width=dm.train_dims[0], vocab_length=dm.vocab_size)

    # train
    trainer.fit(model, dm)

-----------------

Logging/progress bar
--------------------

|

.. image:: /_images/mnist_imgs/mnist_tb.png
    :width: 300
    :align: center
    :alt: Example TB logs

|

Lightning has built-in logging to any of the supported loggers or progress bar.

Log in train loop
^^^^^^^^^^^^^^^^^
To log from the training loop use the `log` method in the `TrainResult`.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        loss = ...
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

The `TrainResult` gives you options for logging on every step and/or at the end of the epoch.
It also allows logging to the progress bar.

.. code-block:: python

        # equivalent
        result.log('train_loss', loss)
        result.log('train_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)

Then boot up your logger or tensorboard instance to view training logs

.. code-block:: bash

    tensorboard --logdir ./lightning_logs

.. warning:: Refreshing the progress bar too frequently in Jupyter notebooks or Colab may freeze your UI.
    We recommend you set `Trainer(progress_bar_refresh_rate=10)`

Log in Val/Test loop
^^^^^^^^^^^^^^^^^^^^
To log from the validation or test loop use the `EvalResult`.

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        loss = ...
        result = pl.EvalResult()
        result.log_dict({'val_loss': loss, 'val_acc': acc})
        return result

Log to the progress bar
^^^^^^^^^^^^^^^^^^^^^^^
|

.. code-block:: shell

    Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]

|

In addition to visual logging, you can log to the progress bar by setting `prog_bar` to True

.. code-block:: python

    def training_step(self, batch, batch_idx):
        loss = ...
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)

-----------------

Advanced loop aggregation
-------------------------
For certain train/val/test loops, you may wish to do more than just logging. In this case,
you can also implement `__epoch_end` which gives you the output for each step

Here's the motivating Pytorch example:

.. code-block:: python

    validation_step_outputs = []
    for batch_idx, batch in val_dataloader():
        out = validation_step(batch, batch_idx)
        validation_step_outputs.append(out)

    validation_epoch_end(validation_step_outputs)

And the lightning equivalent

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        loss = ...
        predictions = ...
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        result.predictions = predictions

     def validation_epoch_end(self, validation_step_outputs):
        all_val_losses = validation_step_outputs.val_loss
        all_predictions = validation_step_outputs.predictions

Why do you need Lightning?
--------------------------
The MAIN teakeaway points are:

- Lightning is for professional AI researchers/production teams.
- Lightning is organized PyTorch. It is not an abstraction.
- You STILL keep pure PyTorch.
- You DON't lose any flexibility.
- You can get rid of all of your boilerplate.
- You make your code generalizable to any hardware.
- Your code is now readable and easier to reproduce (ie: you help with the reproducibility crisis).
- Your LightningModule is still just a pure PyTorch module.

Lightning is for you if
^^^^^^^^^^^^^^^^^^^^^^^

- You're a professional researcher/ml engineer working on non-trivial deep learning.
- You already know PyTorch and are not a beginner.
- You want to iterate through research much faster.
- You want to put models into production much faster.
- You need full control of all the details but don't need the boilerplate.
- You want to leverage code written by hundreds of AI researchers, research engs and PhDs from the world's top AI labs.
- You need GPUs, multi-node training, half-precision and TPUs.
- You want research code that is rigorously tested (500+ tests) across CPUs/multi-GPUs/multi-TPUs on every pull-request.

Some more cool features
^^^^^^^^^^^^^^^^^^^^^^^
Here are (some) of the other things you can do with lightning:

- Automatic checkpointing.
- Automatic early stopping.
- Automatically overfit your model for a sanity test.
- Automatic truncated-back-propagation-through-time.
- Automatically scale your batch size.
- Automatically attempt to find a good learning rate.
- Add arbitrary callbacks
- Hit every line of your code once to see if you have bugs (instead of waiting hours to crash on validation ;)
- Load checkpoints directly from S3.
- Move from CPUs to GPUs or TPUs without code changes.
- Profile your code for speed/memory bottlenecks.
- Scale to massive compute clusters.
- Use multiple dataloaders per train/val/test loop.
- Use multiple optimizers to do Reinforcement learning or even GANs.

Example:
^^^^^^^^
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

Masterclass
-----------
You can learn Lightning in-depth by watching our Masterclass.

.. image:: _images/general/PTL101_youtube_thumbnail.jpg
    :width: 500
    :align: center
    :alt: Masterclass
    :target: https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2
