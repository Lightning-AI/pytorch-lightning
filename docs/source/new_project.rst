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
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def training_step(self, batch, batch_idx):
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
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
    model = LitAutoEncoder()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer()
    trainer.fit(model, train_loader)

-------------

**********
Use/Deploy
**********
When you're done training, export to your favorite format or use for predictions.

If you want to use a LightningModule for predictions, you have 2 options.

Option 1: Pull out the relevant parts you need for prediction

.. code-block:: python

    # ----------------------------------
    # to use as embedding extractor
    # ----------------------------------
    autoencoder = LitAutoEncoder.load_from_checkpoint('path/to/checkpoint_file.ckpt')
    model = autoencoder.encoder
    model.eval()

    # ----------------------------------
    # to use as image generator
    # ----------------------------------
    model = autoencoder.decoder
    model.eval()

Option 2: Add a forward method to enable predictions however you want.

.. code-block:: python

    # ----------------------------------
    # using the AE to extract embeddings
    # ----------------------------------
    class LitAutoEncoder(pl.LightningModule):
        def forward(self, x):
            embedding = self.encoder(x)

    autoencoder = LitAutoencoder()
    autoencoder = autoencoder(torch.rand(1, 28 * 28))

    # ----------------------------------
    # or using the AE to generate images
    # ----------------------------------
    class LitAutoEncoder(pl.LightningModule):
        def forward(self):
            z = torch.rand(1, 28 * 28)
            image = self.decoder(z)
            image = image.view(1, 1, 28, 28)
            return image

    autoencoder = LitAutoencoder()
    image_sample = autoencoder(()

Option 3: Or for a production system

.. code-block:: python

    # ----------------------------------
    # torchscript
    # ----------------------------------
    model = LitAutoEncoder()
    torch.jit.save(model.to_torchscript(), "model.pt")
    os.path.isfile("model.pt")

    # ----------------------------------
    # onnx
    # ----------------------------------
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
         model = LitAutoEncoder()
         input_sample = torch.randn((1, 28 * 28))
         model.to_onnx(tmpfile.name, input_sample, export_params=True)
         os.path.isfile(tmpfile.name)

-----------

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
Although you can return a simple tensor for the loss, if you want to log to the progress bar,
to tensorboard or your favorite library, you can use the
:class:`~pytorch_lightning.core.step_result.TrainResult` and :class:`~pytorch_lightning.core.step_result.EvalResult`
objects.

These objects are just plain Dictionaries but error check for you to avoid things like memory leaks and automatically
syncs metrics across GPUs/TPUs so you don't have to.

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
A callback is an arbitrary self-contained program that can be executed at arbitrary parts of the training loop.

Things you can do with a callback:

- send emails at some point in training
- grow the model
- update learning rates
- visualize gradients
- ...
- you are limited by your imagination

Here's an example adding a not-so-fancy learning rate decay rule:

.. code-block:: python

    class DecayLearningRate(pl.Callback)

        def __init__(self):
            self.old_lrs = []

        def on_train_start(self, trainer, pl_module):
            # track the initial learning rates
            for opt_idx in optimizer in enumerate(trainer.optimizers):
                group = []
                for param_group in optimizer.param_groups:
                    group.append(param_group['lr'])
                self.old_lrs.append(group)

        def on_train_epoch_end(self, trainer, pl_module):
            for opt_idx in optimizer in enumerate(trainer.optimizers):
                old_lr_group = self.old_lrs[opt_idx]
                new_lr_group = []
                for p_idx, param_group in enumerate(optimizer.param_groups):
                    old_lr = old_lr_group[p_idx]
                    new_lr = old_lr * 0.98
                    new_lr_group.append(new_lr)
                    param_group['lr'] = new_lr
                 self.old_lrs[opt_idx] = new_lr_group

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
