:orphan:

###################
Level 1: Start here
###################
PyTorch Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Full flexibility
   :description: Try any ideas using raw PyTorch without the boilerplate.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
   :height: 225

.. displayitem::
   :description: By decoupling research code from engineering, your code becomes readable.
   :header: Readability
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_no_boilerplate.png
   :height: 225

.. displayitem::
   :description: Change between GPU/TPU/HPU etc... without code changes.
   :header: Use any hardware
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_hardware.png
   :height: 225

.. displayitem::
   :description: Lightning offers a minimal API which can be learned very quickly.
   :header: Adopt in 15 minutes
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_fast.png
   :height: 225

.. displayitem::
   :description: We've done all the testing so you don't have to.
   :header: Built-in testing
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_testing.png
   :height: 225

.. displayitem::
   :description: Always replicate results reliably
   :header: Fully reproducible
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_reproducible.png
   :height: 225


.. raw:: html

        </div>
    </div>

.. End of callout item section

----

Lightning vastly simplifies deep learning code

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_docs_animation_final.m4v"></video>

----

In this guide we’ll show you how to organize your PyTorch code into Lightning in 2 steps.

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


*********************************
Step 0: Install PyTorch Lightning
*********************************
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


******************************
Step 1: Define LightningModule
******************************

.. testcode::

    class LitAutoEncoder(pl.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

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

    # assemble your LightningModule
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    autoencoder = LitAutoEncoder(encoder, decoder)

----------

*****************************
Step 2: Fit Lightning Trainer
*****************************

First, define the data however you want. Lightning just needs a :class:`~torch.utils.data.DataLoader` for the train/val/test/predict splits.

.. code-block:: python

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

Next, init the :doc:`LightningModule <../common/lightning_module>` and the PyTorch Lightning :doc:`Trainer <../common/trainer>`,
then call fit with both the data and model.

.. code-block:: python

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

----

************************
Expert-level Flexibility
************************

Lightning has 3 primary mechanisms to enable full flexibility

Hooks
=====

Customize any part of training (such as the backward pass) by overriding any
of the 20+ hooks found in :ref:`lightning_hooks`

.. testcode::

    class LitAutoEncoder(pl.LightningModule):
        def backward(self, loss, optimizer, optimizer_idx):
            loss.backward()

Trainer flags
=============

Training tips/tricks, custom cluster integrations or even the latest SOTA techniques can be enabled via the Lightning Trainer.

.. code::

   # train 1TB+ parameter models with Deepspeed/fsdp
   trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2", precision=16)

   # helpful flags for rapid idea iteration
   trainer = Trainer(max_epochs=10, min_epochs=5, overfit_batches=1)

   # and even the latest state of the art techniques
   trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])

Callbacks
=========

Write arbitrary modular code that can run during the lifecycle of your model

.. code::

   trainer = Trainer(callbacks=[DeviceStatsMonitor()])

Own your loop
=============

For certain types of work at the bleeding-edge of research, Lightning offers experts full control of their training loops in various ways.

- `Lightning Lite <lightning_lite.html>`_
- `Manual optimization <../common/optimization.html#manual-optimization>`_
- `Loops <../extensions/loops.html?highlight=loops>`_
