Research projects tend to test different approaches to the same dataset.
This is very easy to do in Lightning with inheritance.

For example, imagine we now want to train an ``AutoEncoder`` to use as a feature extractor for MNIST images.
The only things that change in the ``AutoEncoder`` model are the init, forward, training, validation and test step.

.. code-block:: python

    class Encoder(torch.nn.Module):
        pass


    class Decoder(torch.nn.Module):
        pass


    class AutoEncoder(LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.metric = MSE()

        def forward(self, x):
            return self.encoder(x)

        def training_step(self, batch, batch_idx):
            x, _ = batch

            representation = self.encoder(x)
            x_hat = self.decoder(representation)

            loss = self.metric(x, x_hat)
            return loss

        def validation_step(self, batch, batch_idx):
            self._shared_eval(batch, batch_idx, "val")

        def test_step(self, batch, batch_idx):
            self._shared_eval(batch, batch_idx, "test")

        def _shared_eval(self, batch, batch_idx, prefix):
            x, _ = batch
            representation = self.encoder(x)
            x_hat = self.decoder(representation)

            loss = self.metric(x, x_hat)
            self.log(f"{prefix}_loss", loss)


and we can train this using the same Trainer instance:

.. code-block:: python

    autoencoder = AutoEncoder()
    trainer = Trainer()
    trainer.fit(autoencoder, train_dataloader, val_dataloader)

And remember that the forward method should define the practical use of a LightningModule.
In this case, we want to use the `AutoEncoder` to extract image representations

.. code-block:: python

    some_images = torch.Tensor(32, 1, 28, 28)
    representations = autoencoder(some_images)
