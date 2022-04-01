:orphan:

Introductory Level 1
====================
In this level you'll learn how to run any PyTorch nn.Module with a LightningModule and the Lightning Trainer

.. note:: Don't know PyTorch yet? `Read this short tutorial <notebooks/course_UvA-DL/01-introduction-to-pytorch.html>`_.

Organizing your code with PyTorch Lightning allows your code to:

* Keep all the flexibility (this is all pure PyTorch), but removes a ton of boilerplate
* More readable by decoupling the research code from the engineering
* Easier to reproduce
* Less error-prone by automating most of the training loop and tricky engineering
* Scalable to any hardware without changing your model

----

.. testsetup:: *

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    import pytorch_lightning as pl
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer

.. _new_project:


Step 0: Install PyTorch Lightning
=================================
.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>

For `pip <https://pypi.org/project/pytorch-lightning/>`_ (and conda) users

.. code-block:: bash

    pip install pytorch-lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

Or directly from `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

.. raw:: html

      </div>
   </div>

Or read the `advanced install guide <starter/installation.html>`_

----------

Import the following:

.. testcode::
    :skipif: not _TORCHVISION_AVAILABLE

    import os
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, random_split
    import pytorch_lightning as pl


Step 1: Define LightningModule
==============================

.. testcode::

    class LitAutoEncoder(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        def forward(self, x):
            # in lightning, forward defines the prediction/inference actions
            embedding = self.encoder(x)
            return embedding

        def training_step(self, batch, batch_idx):
            # training_step defined the train loop.
            # It is independent of forward
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer


**SYSTEM VS MODEL**

A :doc:`lightning module <../common/lightning_module>` defines a *system* not just a model.

.. figure:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/model_system.png
    :width: 400

Examples of systems are:

- `Autoencoder <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/autoencoder.py>`_
- `BERT <https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/text-transformers.ipynb>`_
- `DQN <https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/reinforce-learning-DQN.ipynb>`_
- `GAN <https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/basic-gan.ipynb>`_
- `Image classifier <https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/mnist-hello-world.ipynb>`_
- `Semantic Segmentation <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/semantic_segmentation.py>`_
- `and a lot more <https://github.com/PyTorchLightning/lightning-tutorials/tree/publication/.notebooks/lightning_examples>`_

Under the hood, a LightningModule is still just a :class:`torch.nn.Module` that groups all research code into a single file to make it self-contained:

- The Train loop
- The Validation loop
- The Test loop
- The Prediction loop
- The Model or system of Models
- The Optimizers and LR Schedulers

You can customize any part of training (such as the backward pass) by overriding any
of the 20+ hooks found in :ref:`lightning_hooks`

.. testcode::

    class LitAutoEncoder(pl.LightningModule):
        def backward(self, loss, optimizer, optimizer_idx):
            loss.backward()

**FORWARD vs TRAINING_STEP**

In Lightning we suggest separating training from inference. The ``training_step`` defines
the full training loop. We encourage users to use the ``forward`` to define inference actions.

For example, in this case we can define the autoencoder to act as an embedding extractor:

.. code-block:: python

    def forward(self, batch):
        embeddings = self.encoder(batch)
        return embeddings

Of course, nothing is preventing you from using ``forward`` from within the ``training_step``.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        ...
        embeddings = self.encoder(batch)
        output = self.decoder(embeddings)

It really comes down to your application. We do, however, recommend that you keep both intents separate.

* Use ``forward`` for inference (predicting).
* Use ``training_step`` for training.

More details in :doc:`LightningModule <../common/lightning_module>` docs.

----------

Step 2: Fit with Lightning Trainer
==================================

First, define the data however you want. Lightning just needs a :class:`~torch.utils.data.DataLoader` for the train/val/test/predict splits.

.. code-block:: python

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

Next, init the :doc:`LightningModule <../common/lightning_module>` and the PyTorch Lightning :doc:`Trainer <../common/trainer>`,
then call fit with both the data and model.

.. code-block:: python

    # init model
    autoencoder = LitAutoEncoder()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(accelerator="gpu", devices=8) (if you have GPUs)
    trainer = pl.Trainer()
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

The :class:`~pytorch_lightning.trainer.Trainer` automates:

* Epoch and batch iteration
* ``optimizer.step()``, ``loss.backward()``, ``optimizer.zero_grad()`` calls
* Calling of ``model.eval()``, enabling/disabling grads during evaluation
* :doc:`Checkpoint Saving and Loading <../common/checkpointing>`
* Tensorboard (see :doc:`loggers <../common/loggers>` options)
* :ref:`Multi-GPU <accelerators/gpu:Multi GPU Training>` support
* :doc:`TPU <../accelerators/tpu>`
* :ref:`16-bit precision AMP <amp>` support

.. tip:: If you prefer to manually manage optimizers, you can use the :ref:`manual_opt` mode (i.e., RL, GANs, and so on).


**That's it!**

These are the main two components you need to know in Lightning in general. All the other features of Lightning are either
features of the Trainer or LightningModule or are extensions for advanced use-cases.



----------

Here's a 3 minute conversion guide for PyTorch projects:

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_docs_animation_final.m4v"></video>

----------