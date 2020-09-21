.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.trainer.trainer import Trainer
    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from torch.utils.data import random_split

.. _new_project:

####################
Lightning in 2 steps
####################

**In this guide we'll show you how to organize your PyTorch code into Lightning in 2 steps.**

Organizing your code with PyTorch Lightning makes your code:

* Keep all the flexibility (this is all pure PyTorch), but removes a ton of boilerplate
* More readable by decoupling the research code from the engineering
* Easier to reproduce
* Less error prone by automating most of the training loop and tricky engineering
* Scalable to any hardware without changing your model

----------

Here's a 2 minute conversion guide for PyTorch projects:

.. raw:: html

    <video width="100%" controls autoplay muted playsinline src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_quick_start_full.m4v"></video>

----------

*********************************
Step 0: Install PyTorch Lightning
*********************************


You can install using `pip <https://pypi.org/project/pytorch-lightning/>`_

.. code-block:: bash

    pip install pytorch-lightning

Or with `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_ (see how to install conda `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_):

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

You could also use conda environments

.. code-block:: bash

    conda activate my_env
    pip install pytorch-lightning

----------

Import the following:

.. code-block:: python

    import os
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from torch.utils.data import random_split

******************************
Step 1: Define LightningModule
******************************

.. code-block::


    class LitAutoEncoder(pl.LightningModule):

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 11))
            self.decoder = nn.Sequential(nn.Linear(11, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def forward(self, x):
            # for inference we want to extract embeddings
            return self.encoder(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer


A :class:`~pytorch_lightning.core.LightningModule` defines a system such as a GAN, VAE or MNIST classifier.
It is a :class:`torch.nn.Module` that groups all research code into a single file to make it self-contained:

- The Train loop
- The Validation loop
- The Test loop
- The Model + system architecture
- The Optimizer

You can customize any part of training (such as the backward pass) by overriding any
of the 20+ hooks found in :ref:`hooks`

.. code-block:: python

    class LitModel(pl.LightningModule):

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

When you're done training, export to your favorite format or use for predictions


.. code-block:: python

    # use as regular nn.Module
    model = LitAutoEncoder()
    image = torch.rand(1, 28 * 28)
    embedding = model(image)

    # onnx
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
         model = LitAutoEncoder()
         input_sample = torch.randn((1, 28 * 28))
         model.to_onnx(tmpfile.name, input_sample, export_params=True)
         os.path.isfile(tmpfile.name)

    # torchscript
    model = LitAutoEncoder()
    torch.jit.save(model.to_torchscript(), "model.pt")
    os.path.isfile("model.pt")

More details in :ref:`lightning_module` docs.


----------

**************************
Step 2: Fit with a Trainer
**************************

First, define the data in whatever way you want. Lightning just needs a dataloader per split you might want.

.. code-block:: python

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

.. code-block:: python

    # init model
    model = LitModel()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer()
    trainer.fit(model, train_loader)

-----

***********
Checkpoints
***********
Once you've trained, you can load the checkpoints as follows:

.. code-block:: python

    model = LitModel.load_from_checkpoint(path)

The above checkpoint knows all the arguments needed to init the model and set the state dict.
If you prefer to do it manually, here's the equivalent

.. code-block:: python

    # load the ckpt
    ckpt = torch.load('path/to/checkpoint.ckpt')

    # equivalent to the above
    model = LitModel()
    model.load_state_dict(ckpt['state_dict'])

--------

*****************
Optional features
*****************

TrainResult/EvalResult
======================
Instead of returning the loss you can also use :class:`~pytorch_lightning.core.step_result.TrainResult` and :class:`~pytorch_lightning.core.step_result.EvalResult`, plain Dict objects that give you options for logging on every step and/or at the end of the epoch.
It also allows logging to the progress bar (by setting prog_bar=True). Read more in :ref:`results`.

.. code-block::

    class LitModel(pl.LightningModule):

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = pl.TrainResult(minimize=loss)
            # Add logging to progress bar (note that refreshing the progress bar too frequently
            # in Jupyter notebooks or Colab may freeze your UI) 
            result.log('train_loss', loss, prog_bar=True)
            return result
            
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            # Checkpoint model based on validation loss
            result = pl.EvalResult(checkpoint_on=loss)
            result.log('val_loss', loss)
            return result

            
Callbacks
=========
A :class:`~pytorch_lightning.core.LightningModule` handles advances cases by allowing you to override any critical part of training
via :ref:`hooks` that are called on your :class:`~pytorch_lightning.core.LightningModule`.

.. code-block::

    class LitModel(pl.LightningModule):

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()
            
        def optimizer_step(self, epoch, batch_idx,
                           optimizer, optimizer_idx,
                           second_order_closure,
                           on_tpu, using_native_amp, using_lbfgs):
            optimizer.step()
            
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

.. code-block::

    class LitModel(pl.LightningModule):
    
        def validation_step(self, batch, batch_idx):
            loss = ...
            predictions = ...
            result = pl.EvalResult(checkpoint_on=loss)
            result.log('val_loss', loss)
            result.predictions = predictions

         def validation_epoch_end(self, validation_step_outputs):
            all_val_losses = validation_step_outputs.val_loss
            all_predictions = validation_step_outputs.predictions

Datamodules
===========
DataLoader and data processing code tends to end up scattered around.
Make your data code more reusable by organizing
it into a :class:`~pytorch_lightning.core.datamodule.LightningDataModule`

.. code-block:: python

  class MNISTDataModule(pl.LightningDataModule):

        def __init__(self, batch_size=32):
            super().__init__()
            self.batch_size = batch_size

        # When doing distributed training, Datamodules have two optional arguments for
        # granular control over download/prepare/splitting data:

        # OPTIONAL, called only on 1 GPU/machine
        def prepare_data(self):
            MNIST(os.getcwd(), train=True, download=True)
            MNIST(os.getcwd(), train=False, download=True)

        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        def setup(self, stage):
            # transforms
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # split dataset
            if stage == 'fit':
                mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
                self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
            if stage == 'test':
                self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

        # return the dataloader for each split
        def train_dataloader(self):
            mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
            return mnist_train

        def val_dataloader(self):
            mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
            return mnist_val

        def test_dataloader(self):
            mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
            return mnist_test

:class:`~pytorch_lightning.core.datamodule.LightningDataModule` is designed to enable sharing and reusing data splits
and transforms across different projects. It encapsulates all the steps needed to process data: downloading,
tokenizing, processing etc.

Now you can simply pass your :class:`~pytorch_lightning.core.datamodule.LightningDataModule` to
the :class:`~pytorch_lightning.trainer.Trainer`:

.. code-block::

    # init model
    model = LitModel()

    # init data
    dm = MNISTDataModule()

    # train
    trainer = pl.Trainer()
    trainer.fit(model, dm)

    # test
    trainer.test(datamodule=dm)

DataModules are specifically useful for building models based on data. Read more on :ref:`datamodules`.

----------

********************
Using CPUs/GPUs/TPUs
********************
It's trivial to use CPUs, GPUs or TPUs in Lightning. There's NO NEED to change your code, simply change the :class:`~pytorch_lightning.trainer.Trainer` options.

.. code-block:: python

    # train on CPU
    trainer = pl.Trainer()

    # train on 8 CPUs
    trainer = pl.Trainer(num_processes=8)

    # train on 1 GPU
    trainer = pl.Trainer(gpus=1)

    # train on the ith GPU
    trainer = pl.Trainer(gpus=[3])

    # train on multiple GPUs
    trainer = pl.Trainer(gpus=3)

    # train on multiple GPUs across nodes (32 gpus here)
    trainer = pl.Trainer(gpus=4, num_nodes=8)

    # train on gpu 1, 3, 5 (3 gpus total)
    trainer = pl.Trainer(gpus=[1, 3, 5])

    # train on 8 GPU cores
    trainer = pl.Trainer(tpu_cores=8)

------

*********
Debugging
*********
Lightning has many tools for debugging:

.. code-block:: python

    # use only 10 train batches and 3 val batches
    Trainer(limit_train_batches=10, limit_val_batches=3)

    # overfit the same batch
    Trainer(overfit_batches=1)

    # unit test all the code (check every line)
    Trainer(fast_dev_run=True)

    # train only 20% of an epoch
    Trainer(limit_train_batches=0.2)

    # run validation every 20% of a training epoch
    Trainer(val_check_interval=0.2)

    # find bottlenecks
    Trainer(profiler=True)

    # ... and 20+ more tools

**********
Learn more
**********

That's it! Once you build your module, data, and call trainer.fit(), Lightning trainer calls each loop at the correct time as needed.

You can then boot up your logger or tensorboard instance to view training logs

.. code-block:: bash

    tensorboard --logdir ./lightning_logs
 
---------------


Advanced Lightning Features
===========================

Once you define and train your first Lightning model, you might want to try other cool features like

- :ref:`loggers`
- :ref:`Automatic checkpointing <weights_loading>`
- :ref:`Automatic early stopping <early_stopping>`
- :ref:`Add custom callbacks <callbacks>` (self-contained programs that can be reused across projects)
- :ref:`Dry run mode <debugging:fast_dev_run>` (Hit every line of your code once to see if you have bugs, instead of waiting hours to crash on validation :)
- :ref:`Automatically overfit your model for a sanity test <debugging:Make model overfit on subset of data>`
- :ref:`Automatic truncated-back-propagation-through-time <trainer:truncated_bptt_steps>`
- :ref:`Automatically scale your batch size <training_tricks:Auto scaling of batch size>`
- :ref:`Automatically find a good learning rate <lr_finder>`
- :ref:`Load checkpoints directly from S3 <weights_loading:Checkpoint Loading>`
- :ref:`Profile your code for speed/memory bottlenecks <profiler>`
- :ref:`Scale to massive compute clusters <slurm>`
- :ref:`Use multiple dataloaders per train/val/test loop <multiple_loaders>`
- :ref:`Use multiple optimizers to do Reinforcement learning or even GANs <optimizers:Use multiple optimizers (like GANs)>`

Or read our :ref:`introduction_guide` to learn more!

-------------

Masterclass
===========

Go pro by tunning in to our Masterclass! New episodes every week.

.. image:: _images/general/PTL101_youtube_thumbnail.jpg
    :width: 500
    :align: center
    :alt: Masterclass
    :target: https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2
