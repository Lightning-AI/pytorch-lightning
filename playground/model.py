import os

import torch
import torchvision
import pytorch_lightning as pl


class CoolModel(pl.LightningModule):
    def __init__(self, hparams):
        super(CoolModel, self).__init__()
        self.hparams = hparams
        if hparams.__class__.__name__ == 'Namespace':
            print('hparam_type: Namespace')
            self.net = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, hparams.n_filters),
                torch.nn.ReLU(),
                torch.nn.Linear(hparams.n_filters, 10),
                torch.nn.LogSoftmax(dim=1)
            )
        elif hparams.__class__.__name__ == 'dict':
            print('hparam_type: dict')
            self.net = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, hparams['n_filters']),
                torch.nn.ReLU(),
                torch.nn.Linear(hparams['n_filters'], 10),
                torch.nn.LogSoftmax(dim=1)
            )

    def forward(self, x):
        batch_size, *_ = x.size()
        return self.net(x.view(batch_size, -1))

    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = torchvision.datasets.MNIST(
            os.getcwd(), train=True, download=True, transform=transform)
        mnist_train = torch.utils.data.Subset(mnist, list(range(55000)))
        return torch.utils.data.DataLoader(mnist_train, batch_size=64, drop_last=True)

    def val_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = torchvision.datasets.MNIST(
            os.getcwd(), train=True, download=True, transform=transform)
        mnist_val = torch.utils.data.Subset(mnist, list(range(55000)))
        return torch.utils.data.DataLoader(mnist_val, batch_size=64, drop_last=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.nll_loss(logits, y)
        return {'loss': loss, 'log': {'loss/train': loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.nll_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss, 'log': {'loss/val': val_loss}}