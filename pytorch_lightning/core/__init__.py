"""
A LightningModule is a strict superclass of torch.nn.Module but provides an interface to standardize
the "ingredients" for a research or production system.

- The model/system definition (__init__)
- The model/system computations (forward)
- What happens in the training loop (training_step, training_end)
- What happens in the validation loop (validation_step, validation_end)
- What happens in the test loop (test_step, test_end)
- What optimizers to use (configure_optimizers)
- What data to use (train_dataloader, val_dataloader, test_dataloader)

Most methods are optional. Here's a minimal example.

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    import pytorch_lightning as pl

    class CoolModel(pl.LightningModule):

        def __init__(self):
            super(CoolModel, self).__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.forward(x)
            return {'loss': F.cross_entropy(y_hat, y)}

        def validation_step(self, batch, batch_idx):
            # OPTIONAL
            x, y = batch
            y_hat = self.forward(x)
            return {'val_loss': F.cross_entropy(y_hat, y)}

        def validation_end(self, outputs):
            # OPTIONAL
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            return {'val_loss': val_loss_mean}

        def test_step(self, batch, batch_idx):
            # OPTIONAL
            x, y = batch
            y_hat = self.forward(x)
            return {'test_loss': F.cross_entropy(y_hat, y)}

        def test_end(self, outputs):
            # OPTIONAL
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            return {'test_loss': test_loss_mean}

        def configure_optimizers(self):
            # REQUIRED
            return torch.optim.Adam(self.parameters(), lr=0.02)

        @pl.data_loader
        def train_dataloader(self):
            return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                              transform=transforms.ToTensor()), batch_size=32)

        @pl.data_loader
        def val_dataloader(self):
            # OPTIONAL
            # can also return a list of val dataloaders
            return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                              transform=transforms.ToTensor()), batch_size=32)

        @pl.data_loader
        def test_dataloader(self):
            # OPTIONAL
            # can also return a list of test dataloaders
            return DataLoader(MNIST(os.getcwd(), train=False, download=True,
                              transform=transforms.ToTensor()), batch_size=32)

Once you've defined the LightningModule, fit  it using a trainer.

.. code-block:: python

   trainer = pl.Trainer()
   model = CoolModel()

   trainer.fit(model)

Check out this
`COLAB <https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg>`_
for a live demo.

"""

from .decorators import data_loader
from .lightning import LightningModule

__all__ = ['LightningModule', 'data_loader']
# __call__ = __all__
