.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader


Quick Start
===========

New to PyTorch Lightning and want to give it a go? Quick prototyping is our specility- you came to the right place!

Organizing your code with PyTorch Lightning makes your code:

* more readable by decoupling the research code from the engineering
* Easier to reproduce
* less error prone by automtaing most of the training loop and tricky engineering
* keep all the flexibility (this is all pure PyTorch), but removes a ton of boilerplate
* scalable to any hardware without changing a thing!

To illustrate, here's the typical PyTorch project structure organized in a LightningModule.

.. figure:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pt_animation_gif.gif
   :alt: Convert from PyTorch to Lightning
   
   
In this guide we will go over how to build your Lightning code in 5 steps!


----------

Step 1: Install PyTorch Lightning
---------------------------------

We reccoment using conda environments

.. code-block:: bash

    conda activate my_env
    pip install pytorch-lightning

Otherwise you can install With pip

.. code-block:: bash

    pip install pytorch-lightning

Or with conda

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

----------

Step 2: Define LightningModule
------------------------------
The :class:`~pytorch_lightning.core.LightningModule` holds your research code:

- The Train loop
- The Validation loop
- The Test loop
- The Model + system architecture
- The Optimizer

A :class:`~pytorch_lightning.core.LightningModule` is a :class:`torch.nn.Module` but with added functionality. It organizes your research code into `hooks<https://pytorch-lightning.readthedocs.io/en/stable/hooks.html>`_


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
            
            
You can use your :class:`~pytorch_lightning.core.LightningModule` just like a PyTorch model. Read more `here<https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html>`_.

----------

Step 3: Define you data loaders
-------------------------------

:class:`~pytorch_lightning.core.DataModule`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A :class:`~pytorch_lightning.core.DataModule` is simply a collection of all 3 data splits but
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


DataModules are specifically useful for building models based on data. Read more `here<https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html>`_.

PyTorch DataLoader
^^^^^^^^^^^^^^^^^^
If you don't want to craete a DataModule you can use plain PyTorch :class:`~torch.utils.data.DataLoader`, and add them to your :class:`~pytorch_lightning.core.lightningModule`:

.. code-block:: python

    class LitModel(pl.LightningModule):

        def train_dataloader(self):
            # your train transforms
            data_set = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
            return DataLoader(data_set)
            
or just use them directly in your trainer.

----------

Step 4: Fit with a Trainer
--------------------------
You init your :class:`~pytorch_lightning.core.LightningModule`, your :class:`~pytorch_lightning.core.DataModule` and then the :class:`~pytorch_lightning.core.Trainer`. 
The Lightning :class:`~pytorch_lightning.core.Trainer` is where the magic happens! It abstracts the most of the training engineering code such as:

* The batch iteration
* The epoch iteration
* Calling of optimizer.step()


.. code-block:: python

    # init model
    model = LitModel()
    # init data
    data_module = MyDataModule()

    # most basic trainer, uses good defaults and data module
    trainer = pl.Trainer()
    trainer.fit(model, data_module)

Using GPUs/TPUs
^^^^^^^^^^^^^^^
It's trivial to use GPUs or TPUs in Lightning. There's NO NEED to change your code, simply change the Trainer options.

.. code-block:: python

    # train on 1, 2, 4, n GPUs
    trainer = pl.Trainer(gpus=1)
    trainer = pl.Trainer(gpus=2)
    trainer = pl.Trainer(gpus=8, num_nodes=n)

    # train on TPUs
    trainer = pl.Trainer(tpu_cores=8)
    trainer = pl.Trainer(tpu_cores=128)

    # even half precision
    trainer = pl.Trainer(gpus=2, precision=16)
    
The :class:`~pytorch_lightning.core.Trainer` will provide

* Automatic checkpoints
* Automatic Tensorboard (or the logger of your choice)
* Automatic CPU/GPU/TPU training
* Automatic 16-bit precision

All of it 100% rigorously tested and benchmarked.

More magical trainer flags can be found `here<https://pytorch-lightning.readthedocs.io/en/stable/trainer.html>`_.

--------------

Training loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Under the hood, lightning does the following (in high-level pseudocode):

.. code-block:: python

    # init model
    model = LitModel()

    # enable training
    torch.set_grad_enabled(True)
    model.train()

    # get data + optimizer
    train_dataloader = model.data_module().train_dataloader()
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


Step 5: Add validation and test loop
------------------------------------

Adding a Validation loop
^^^^^^^^^^^^^^^^^^^^^^^^
To add an (optional) validation loop add the following callback to your :class:`~pytorch_lightning.core.LightningModule`

.. testcode::

    class LitModel(LightningModule):

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)

            result = pl.EvalResult(checkpoint_on=loss)
            result.log('val_loss', loss)
            return result

.. note:: :class:`~pytorch_lightning.core.step_result.EvalResult` is a plain Dict, with convenience functions for logging

And to your :class:`~pytorch_lightning.core.DataModule`:

.. code-block:: python

    class MyDataModule(pl.DataModule):

        def __init__(self):
            ...

        def val_dataloader(self):
            # your val transforms
            return DataLoader(YOUR_DATASET)

And now the trainer will call the validation loop automatically.


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
            for val_batch in model.data_module.val_dataloader:
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
^^^^^^^^^^^^^^^^^^
You might also need an optional test loop. Similarly, add the following callback to your :class:`~pytorch_lightning.core.LightningModule`

.. testcode::

    class LitModel(LightningModule):


        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)

            result = pl.EvalResult()
            result.log('test_loss', loss)
            return result

.. note:: :class:`~pytorch_lightning.core.step_result.EvalResult` is a plain Dict, with convenience functions for logging

And to your :class:`~pytorch_lightning.core.DataModule`:

.. code-block:: python

    class MyDataModule(pl.DataModule):

        def __init__(self):
            ...

        def test_dataloader(self):
            # your val transforms
            return DataLoader(YOUR_DATASET)


However, this time you need to specifically call test (this is done so you don't use the test set by mistake).

.. code-block:: python

    # OPTION 1:
    # test after fit
    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module)


.. code-block:: python

    # OPTION 2:
    # test after loading weights
    model = LitModel.load_from_checkpoint(PATH)
    trainer = Trainer(model, data_module)
    trainer.test(datamodule=data_module)

Test loop under the hood
^^^^^^^^^^^^^^^^^^^^^^^^
Under the hood, lightning does the following in (pseudocode):

.. code-block:: python

    # disable grads + batchnorm + dropout
    torch.set_grad_enabled(False)
    model.eval()

    test_outs = []
    for test_batch in model.data_module.test_dataloader:
        test_out = model.test_step(val_batch)
        test_outs.append(test_out)

    model.test_epoch_end(test_outs)

    # enable grads + batchnorm + dropout
    torch.set_grad_enabled(True)
    model.train()


-----------------

That's it!

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


.. code-block:: python

    # train on 256 GPUs
    trainer = Trainer(
        gpus=8,
        num_nodes=32
    )


.. code-block:: python

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

Once you define and train your first Lightning model, you might want to try other cool features like

- `Logging <https://pytorch-lightning.readthedocs.io/en/stable/loggers.html>`_
- `Automatic checkpointing <https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html>`_
- `Automatic early stopping <https://pytorch-lightning.readthedocs.io/en/stable/early_stopping.html>`_
- `Add custom callbacks <https://pytorch-lightning.readthedocs.io/en/stable/callbacks.html>`_
- `Dry run mode <https://pytorch-lightning.readthedocs.io/en/stable/debugging.html#fast-dev-run>`_ (Hit every line of your code once to see if you have bugs, instead of waiting hours to crash on validation ;)
- `Automatically overfit your model for a sanity test <https://pytorch-lightning.readthedocs.io/en/stable/debugging.html?highlight=overfit#make-model-overfit-on-subset-of-data>`_
- `Automatic truncated-back-propagation-through-time <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.training_loop.html?highlight=truncated#truncated-backpropagation-through-time>`_
- `Automatically scale your batch size <https://pytorch-lightning.readthedocs.io/en/stable/training_tricks.html?highlight=batch%20size#auto-scaling-of-batch-size>`_
- `Automatically find a good learning rate <https://pytorch-lightning.readthedocs.io/en/stable/lr_finder.html>`_
- `Load checkpoints directly from S3 <https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html#checkpoint-loading>`_
- `Profile your code for speed/memory bottlenecks <https://pytorch-lightning.readthedocs.io/en/stable/profiler.html>`_
- `Scale to massive compute clusters <https://pytorch-lightning.readthedocs.io/en/stable/slurm.html>`_
- `Use multiple dataloaders per train/val/test loop <https://pytorch-lightning.readthedocs.io/en/stable/multiple_loaders.html>`_
- `Use multiple optimizers to do Reinforcement learning or even GANs <https://pytorch-lightning.readthedocs.io/en/stable/optimizers.html?highlight=multiple%20optimizers#use-multiple-optimizers-like-gans>`_

-----------

Masterclass
-----------
Go pro by tunning in to our Masterclass! New episodes every week.


.. image:: _images/general/PTL101_youtube_thumbnail.jpg
    :width: 500
    :align: center
    :alt: Masterclass
    :target: https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2
    

Or read our `walkthrough<https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html>`_. to learn more!

