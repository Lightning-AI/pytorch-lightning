Research projects tend to test different approaches to the same dataset.
This is very easy to do in Lightning with inheritance.

For example, imagine we now want to train an ``AutoEncoder`` to use as a feature extractor for images.
The only things that change in the ``LitAutoEncoder`` model are the init, forward, training, validation and test step.

.. code-block:: python

    class Encoder(torch.nn.Module):
        ...


    class Decoder(torch.nn.Module):
        ...


    class AutoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            return self.decoder(self.encoder(x))


    class LitAutoEncoder(LightningModule):
        def __init__(self, auto_encoder):
            super().__init__()
            self.auto_encoder = auto_encoder
            self.metric = torch.nn.MSELoss()

        def forward(self, x):
            return self.auto_encoder.encoder(x)

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x_hat = self.auto_encoder(x)
            loss = self.metric(x, x_hat)
            return loss

        def validation_step(self, batch, batch_idx):
            self._shared_eval(batch, batch_idx, "val")

        def test_step(self, batch, batch_idx):
            self._shared_eval(batch, batch_idx, "test")

        def _shared_eval(self, batch, batch_idx, prefix):
            x, _ = batch
            x_hat = self.auto_encoder(x)
            loss = self.metric(x, x_hat)
            self.log(f"{prefix}_loss", loss)


and we can train this using the ``Trainer``:

.. code-block:: python

    auto_encoder = AutoEncoder()
    lightning_module = LitAutoEncoder(auto_encoder)
    trainer = Trainer()
    trainer.fit(lightning_module, train_dataloader, val_dataloader)

And remember that the forward method should define the practical use of a :class:`~lightning.pytorch.core.LightningModule`.
In this case, we want to use the ``LitAutoEncoder`` to extract image representations:

.. code-block:: python

    some_images = torch.Tensor(32, 1, 28, 28)
    representations = lightning_module(some_images)
