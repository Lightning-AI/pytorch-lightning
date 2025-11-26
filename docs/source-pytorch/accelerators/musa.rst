:orphan:

MUSA training (Advanced)
========================
**Audience:** Users looking to train models on MooreThreads device using MUSA accelerator.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

MUSAAccelerator Overview
--------------------
torch_musa is an extended Python package based on PyTorch that enables full utilization of MooreThreads graphics cards' 
super computing power. Combined with PyTorch, users can take advantage of the strong power of MooreThreads graphics cards 
through torch_musa.

PyTorch Lightning automatically finds these weights and ties them after the modules are moved to the
MUSA device under the hood. It will ensure that the weights among the modules are shared but not copied
independently.


Example:

.. code-block:: python
    import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
    import pytorch_lightning as L

    # Step 1: Define a LightningModule
    class LitAutoEncoder(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def forward(self, x):
            # in lightning, forward defines the prediction/inference actions
            embedding = self.encoder(x)
            return embedding

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop. It is independent of forward
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    def main():
        # -------------------
        # Step 2: Define data
        # -------------------
        dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
        train, val = data.random_split(dataset, [55000, 5000])

        # -------------------
        # Step 3: Train
        # -------------------
        autoencoder = LitAutoEncoder()
        # we also support accelerator="auto" or accelerator="musa"
        trainer = L.Trainer(accelerator="gpu")
        trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))

    if __name__ == '__main__':

        main()
----

MUSA
----
MUSA is the library that interfaces PyTorch with the MooreThreads graphics cards.
For more information check out `MUSA <https://github.com/MooreThreads/torch_musa>`_.
