"""
Lightning Module interface
==========================

A lightning module is a strict superclass of nn.Module, it provides a standard interface
 for the trainer to interact with the model.

The easiest thing to do is copy the minimal example below and modify accordingly.

Otherwise, to Define a Lightning Module, implement the following methods:


Minimal example
---------------

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
            # not the best model...
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            # REQUIRED
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


How do these methods fit into the broader training?
---------------------------------------------------

The LightningModule interface is on the right. Each method corresponds
 to a part of a research project. Lightning automates everything not in blue.

.. figure::  docs/source/_static/images/overview_flat.jpg
   :align:   center

   Overview.


Optional Methods
----------------

**add_model_specific_args**

.. code-block:: python

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir)

Lightning has a list of default argparse commands.
 This method is your chance to add or modify commands specific to your model.
 The `hyperparameter argument parser
  <https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser>`_
 is available anywhere in your model by calling self.hparams.

**Return**
An argument parser

**Example**

.. code-block:: python

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=False)
        parser.add_argument('--in_features', default=28*28)
        parser.add_argument('--out_features', default=10)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.opt_list('--learning_rate', default=0.001, type=float,
                        options=[0.0001, 0.0005, 0.001, 0.005], tunable=False)
        parser.opt_list('--batch_size', default=256, type=int,
                        options=[32, 64, 128, 256], tunable=False)
        parser.opt_list('--optimizer_name', default='adam', type=str,
                        options=['adam'], tunable=False)
        return parser

"""
