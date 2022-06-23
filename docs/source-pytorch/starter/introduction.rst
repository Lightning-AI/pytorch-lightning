:orphan:

#######################
Lightning in 15 minutes
#######################
**Required background:** None

**Goal:** In this guide, we'll walk you through the 7 key steps of a typical Lightning workflow.

PyTorch Lightning is the deep learning framework with "batteries included" for professional AI researchers and machine learning engineers who need maximal flexibility while super-charging performance at scale.

.. join_slack::
   :align: left
   :margin: 20


Lightning organizes PyTorch code to remove boilerplate and unlock scalability.

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_docs_animation_final.m4v"></video>

|

By organizing PyTorch code, lightning enables:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Full flexibility
   :description: Try any ideas using raw PyTorch without the boilerplate.
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_full_control.png
   :height: 290

.. displayitem::
   :description: Decoupled research and engineering code enable reproducibility and better readability.
   :header: Reproducible + Readable
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_no_boilerplate.png
   :height: 290

.. displayitem::
   :description: Use multiple GPUs/TPUs/HPUs etc... without code changes.
   :header: Simple multi-GPU training
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_hardware.png
   :height: 290

.. displayitem::
   :description: We've done all the testing so you don't have to.
   :header: Built-in testing
   :col_css: col-md-3
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/card_testing.png
   :height: 290

.. raw:: html

        </div>
    </div>

.. End of callout item section

----

****************************
1: Install PyTorch Lightning
****************************
.. raw:: html

   <div class="row" style='font-size: 16px'>
      <div class='col-md-6'>

For `pip <https://pypi.org/project/pytorch-lightning/>`_ users

.. code-block:: bash

    pip install pytorch-lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

For `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_ users

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

.. raw:: html

      </div>
   </div>

Or read the `advanced install guide <installation.html>`_

----

.. _new_project:

***************************
2: Define a LightningModule
***************************

A LightningModule enables your PyTorch nn.Module to play together in complex ways inside the training_step (there is also an optional validation_step and test_step).

.. testcode::

    import os
    from torch import optim, nn, utils, Tensor
    from torchvision.datasets import MNIST
    import pytorch_lightning as pl

    # define any number of nn.Modules (or use your current ones)
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # define the LightningModule
    class LitAutoEncoder(pl.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = nn.functional.mse_loss(x_hat, x)
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return optimizer


    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)

----

*******************
3: Define a dataset
*******************

Lightning supports ANY iterable (:class:`~torch.utils.data.DataLoader`, numpy, etc...) for the train/val/test/predict splits.

.. code-block:: python

    # setup data
    dataset = MNIST(os.getcwd(), download=True)
    train_loader = utils.data.DataLoader(dataset)

----

******************
4: Train the model
******************

The Lightning :doc:`Trainer <../common/trainer>` "mixes" any :doc:`LightningModule <../common/lightning_module>` with any dataset and abstracts away all the engineering complexity needed for scale.

.. code-block:: python

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

The Lightning :doc:`Trainer <../common/trainer>` automates `40+ tricks <../common/trainer.html#trainer-flags>`_ including:

* Epoch and batch iteration
* ``optimizer.step()``, ``loss.backward()``, ``optimizer.zero_grad()`` calls
* Calling of ``model.eval()``, enabling/disabling grads during evaluation
* :doc:`Checkpoint Saving and Loading <../common/checkpointing>`
* Tensorboard (see :doc:`loggers <../visualize/loggers>` options)
* :doc:`Multi-GPU <../accelerators/gpu>` support
* :doc:`TPU <../accelerators/tpu>`
* :ref:`16-bit precision AMP <speed-amp>` support

----


****************
5: Use the model
****************
Once you've trained the model you can export to onnx, torchscript and put it into production or simply load the weights and run predictions.

.. code:: python

    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    # embed 4 fake images!
    fake_image_batch = Tensor(4, 28 * 28)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

----

*********************
6: Visualize training
*********************
Lightning comes with a *lot* of batteries included. A helpful one is Tensorboard for visualizing experiments.

Run this on your commandline and open your browser to **http://localhost:6006/**

.. code:: bash

    tensorboard --logdir .

----

***********************
7: Supercharge training
***********************
Enable advanced training features using Trainer arguments. These are state-of-the-art techniques that are automatically integrated into your training loop without changes to your code.

.. code::

   # train on 4 GPUs
   trainer = Trainer(
       devices=4,
       accelerator="gpu",
    )

   # train 1TB+ parameter models with Deepspeed/fsdp
   trainer = Trainer(
       devices=4,
       accelerator="gpu",
       strategy="deepspeed_stage_2",
       precision=16
    )

   # 20+ helpful flags for rapid idea iteration
   trainer = Trainer(
       max_epochs=10,
       min_epochs=5,
       overfit_batches=1
    )

   # access the latest state of the art techniques
   trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])

----

********************
Maximize flexibility
********************
Lightning's core guiding principle is to always provide maximal flexibility **without ever hiding any of the PyTorch**.

Lightning offers 5 *added* degrees of flexibility depending on your project's complexity.

----

Customize training loop
=======================

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/custom_loop.png
    :width: 600
    :alt: Injecting custom code in a training loop

Inject custom code anywhere in the Training loop using any of the 20+ methods (:ref:`lightning_hooks`) available in the LightningModule.

.. testcode::

    class LitAutoEncoder(pl.LightningModule):
        def backward(self, loss, optimizer, optimizer_idx):
            loss.backward()

----

Extend the Trainer
==================

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cb.m4v"></video>

If you have multiple lines of code with similar functionalities, you can use callbacks to easily group them together and toggle all of those lines on or off at the same time.

.. code::

   trainer = Trainer(callbacks=[AWSCheckpoints()])

----

Use a raw PyTorch loop
======================

For certain types of work at the bleeding-edge of research, Lightning offers experts full control of their training loops in various ways.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Manual optimization
   :description: Automated training loop, but you own the optimization steps.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/manual_opt.png
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :image_height: 220px
   :height: 320

.. displayitem::
   :header: Lightning Lite
   :description: Full control over loop for migrating complex PyTorch projects.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lite.png
   :button_link: ../model/build_model_expert.html
   :image_height: 220px
   :height: 320

.. displayitem::
   :header: Loops
   :description: Enable meta-learning, reinforcement learning, GANs with full control.
   :col_css: col-md-4
   :image_center: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/loops.png
   :button_link: ../extensions/loops.html
   :image_height: 220px
   :height: 320

.. raw:: html

        </div>
    </div>

.. End of callout item section

----

**********
Next steps
**********
Depending on your use case, you might want to check one of these out next.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 2: Add a validation and test set
   :description: Add validation and test sets to avoid over/underfitting.
   :button_link: ../levels/basic_level_2.html
   :col_css: col-md-3
   :height: 180
   :tag: basic

.. displayitem::
   :header: See more examples
   :description: See examples across computer vision, NLP, RL, etc...
   :col_css: col-md-3
   :button_link: ../tutorials.html
   :height: 180
   :tag: basic

.. displayitem::
   :header: I need my raw PyTorch Loop
   :description: Expert-level control for researchers working on the bleeding-edge
   :col_css: col-md-3
   :button_link: ../model/build_model_expert.html
   :height: 180
   :tag: expert

.. displayitem::
   :header: Deploy your model
   :description: Learn how to predict or put your model into production
   :col_css: col-md-3
   :button_link: ../deploy/production.html
   :height: 180
   :tag: basic

.. raw:: html

        </div>
    </div>
