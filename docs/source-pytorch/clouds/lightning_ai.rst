:orphan:

#############################################
Run single or multi-node on Lightning Studios
#############################################

**Audience**: Users who don't want to waste time on cluster configuration and maintenance.

`Lightning Studios <https://lightning.ai>`_ is a cloud platform where you can build, train, finetune and deploy models without worrying about infrastructure, cost management, scaling, and other technical headaches.
This guide shows you how easy it is to run a PyTorch Lightning training script across multiple machines on Lightning Studios.


----


*************
Initial Setup
*************

First, create a free `Lightning AI account <https://lightning.ai/>`_.
You get free credits every month you can spend on GPU compute.
To use machines with multiple GPUs or run jobs across machines, you need to be on the `Pro or Teams plan <https://lightning.ai/pricing>`_.


----


***************************************
Launch multi-node training in the cloud
***************************************

**Step 1:** Start a new Studio.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/videos/start-studio-for-mmt.mp4
    :width: 800
    :loop:
    :muted:

|

**Step 2:** Bring your code into the Studio. You can clone a GitHub repo, drag and drop local files, or use the following demo example:

.. collapse:: Code Example

    .. code-block:: python

        import lightning as L
        import torch
        import torch.nn.functional as F
        from lightning.pytorch.demos import Transformer, WikiText2
        from torch.utils.data import DataLoader, random_split


        class LanguageDataModule(L.LightningDataModule):
            def __init__(self, batch_size):
                super().__init__()
                self.batch_size = batch_size
                self.vocab_size = 33278

            def prepare_data(self):
                WikiText2(download=True)

            def setup(self, stage):
                dataset = WikiText2()

                # Split data in to train, val, test
                n = len(dataset)
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [n - 4000, 2000, 2000])

            def train_dataloader(self):
                return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            def val_dataloader(self):
                return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

            def test_dataloader(self):
                return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


        class LanguageModel(L.LightningModule):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size
                self.model = None

            def configure_model(self):
                if self.model is None:
                    self.model = Transformer(vocab_size=self.vocab_size)

            def training_step(self, batch, batch_idx):
                input, target = batch
                output = self.model(input, target)
                loss = F.nll_loss(output, target.view(-1))
                self.log("train_loss", loss)
                return loss

            def validation_step(self, batch, batch_idx):
                input, target = batch
                output = self.model(input, target)
                loss = F.nll_loss(output, target.view(-1))
                self.log("val_loss", loss)
                return loss

            def test_step(self, batch, batch_idx):
                input, target = batch
                output = self.model(input, target)
                loss = F.nll_loss(output, target.view(-1))
                self.log("test_loss", loss)
                return loss

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)


        def main():
            L.seed_everything(42)

            datamodule = LanguageDataModule(batch_size=20)
            model = LanguageModel(datamodule.vocab_size)

            # Trainer
            trainer = L.Trainer(gradient_clip_val=0.25, max_epochs=2, strategy="ddp")
            trainer.fit(model, datamodule=datamodule)
            trainer.test(model, datamodule=datamodule)


        if __name__ == "__main__":
            main()

|

**Step 3:** Remove hardcoded accelerator settings if any and let Lightning automatically set them for you. No other changes are required in your script.

.. code-block:: python

    # These are the defaults
    trainer = L.Trainer(accelerator="auto", devices="auto")

    # DON'T hardcode these, leave them default/auto
    # trainer = L.Trainer(accelerator="cpu", devices=3)

|

**Step 4:** Install dependencies and download all necessary data. Test that your script runs in the Studio first. If it runs in the Studio, it will run in multi-node!

|

**Step 5:** Open the Multi-Machine Training (MMT) app. Type the command to run your script, select the machine type and how many machines you want to launch it on. Click "Run" to start the job.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/lightning-ai-mmt-demo-pl.mp4
    :width: 800
    :loop:
    :muted:

After submitting the job, you will be redirected to a page where you can monitor the machine metrics and logs in real-time.


----


****************************
Bring your own cloud account
****************************

As a `Teams or Enterprise <https://lightning.ai/pricing>`_ customer, you have the option to connect your existing cloud account to Lightning AI.
This gives your organization the ability to keep all compute and data on your own cloud account and your Virtual Private Cloud (VPC).


----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Lightning Studios
    :description: Code together. Prototype. Train. Deploy. Host AI web apps. From your browser - with zero setup.
    :col_css: col-md-4
    :button_link: https://lightning.ai
    :height: 150

.. raw:: html

        </div>
    </div>

|
