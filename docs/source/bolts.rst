Bolts
=====
`PyTorch Lightning Bolts <https://pytorch-lightning-bolts.readthedocs.io/en/latest/>`_, is our official collection
of prebuilt models across many research domains.

.. code-block:: bash

    pip install pytorch-lightning-bolts

In bolts we have:

- A collection of pretrained state-of-the-art models.
- A collection of models designed to bootstrap your research.
- A collection of callbacks, transforms, full datasets.
- All models work on CPUs, TPUs, GPUs and 16-bit precision.

-----------------

Quality control
---------------
The Lightning community builds bolts and contributes them to Bolts.
The lightning team guarantees that contributions are:

- Rigorously Tested (CPUs, GPUs, TPUs).
- Rigorously Documented.
- Standardized via PyTorch Lightning.
- Optimized for speed.
- Checked for correctness.

---------

Example 1: Pretrained, prebuilt models
--------------------------------------

.. code-block:: python

    from pl_bolts.models import VAE, GPT2, ImageGPT, PixelCNN
    from pl_bolts.models.self_supervised import AMDIM, CPCV2, SimCLR, MocoV2
    from pl_bolts.models import LinearRegression, LogisticRegression
    from pl_bolts.models.gans import GAN
    from pl_bolts.callbacks import PrintTableMetricsCallback
    from pl_bolts.datamodules import FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule

------------

Example 2: Extend for faster research
-------------------------------------
Bolts are contributed with benchmarks and continuous-integration tests. This means
you can trust the implementations and use them to bootstrap your research much faster.

.. code-block:: python

    from pl_bolts.models import ImageGPT
    from pl_bolts.self_supervised import SimCLR

    class VideoGPT(ImageGPT):

        def training_step(self, batch, batch_idx):
            x, y = batch
            x = _shape_input(x)

            logits = self.gpt(x)
            simclr_features = self.simclr(x)

            # -----------------
            # do something new with GPT logits + simclr_features
            # -----------------

            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

            logs = {"loss": loss}
            return {"loss": loss, "log": logs}

----------

Example 3: Callbacks
--------------------
We also have a collection of callbacks.

.. code-block:: python

    from pl_bolts.callbacks import PrintTableMetricsCallback
    import pytorch_lightning as pl

    trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])

    # loss│train_loss│val_loss│epoch
    # ──────────────────────────────
    # 2.2541470527648926│2.2541470527648926│2.2158432006835938│0
