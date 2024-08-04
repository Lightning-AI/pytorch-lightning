:orphan:

#####################
Train a model (basic)
#####################
**Audience**: Users who need to train a model without coding their own training loops.

----

***********
Add imports
***********
Add the relevant imports at the top of the file

.. code:: python

    import os
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    import lightning as L

----

*****************************
Define the PyTorch nn.Modules
*****************************

.. code:: python

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):
            return self.l1(x)


    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        def forward(self, x):
            return self.l1(x)

----

************************
Define a LightningModule
************************
The LightningModule is the full **recipe** that defines how your nn.Modules interact.

- The **training_step** defines how the *nn.Modules* interact together.
- In the **configure_optimizers** define the optimizer(s) for your models.

.. code:: python

    class LitAutoEncoder(L.LightningModule):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

----

***************************
Define the training dataset
***************************
Define a PyTorch :class:`~torch.utils.data.DataLoader` which contains your training dataset.

.. code-block:: python

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

----

***************
Train the model
***************
To train the model use the Lightning :doc:`Trainer <../common/trainer>` which handles all the engineering and abstracts away all the complexity needed for scale.

.. code-block:: python

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train model
    trainer = L.Trainer()
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

----

***************************
Eliminate the training loop
***************************
Under the hood, the Lightning Trainer runs the following training loop on your behalf

.. code:: python

    autoencoder = LitAutoEncoder(Encoder(), Decoder())
    optimizer = autoencoder.configure_optimizers()

    for batch_idx, batch in enumerate(train_loader):
        loss = autoencoder.training_step(batch, batch_idx)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

The power of Lightning comes when the training loop gets complicated as you add validation/test splits, schedulers, distributed training and all the latest SOTA techniques.

With Lightning, you can add mix all these techniques together without needing to rewrite a new loop every time.
