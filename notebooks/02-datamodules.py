# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/02-datamodules.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="2O5r7QvP8-rt" colab_type="text"
# # PyTorch Lightning DataModules ⚡
#
# With the release of `pytorch-lightning` version 0.9.0, we have included a new class called `LightningDataModule` to help you decouple data related hooks from your `LightningModule`.
#
# This notebook will walk you through how to start using Datamodules.
#
# The most up to date documentation on datamodules can be found [here](https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html).
#
# ---
#
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)

# + [markdown] id="6RYMhmfA9ATN" colab_type="text"
# ### Setup
# Lightning is easy to install. Simply ```pip install pytorch-lightning```

# + id="lj2zD-wsbvGr" colab_type="code" colab={}
# ! pip install pytorch-lightning --quiet

# + [markdown] id="8g2mbvy-9xDI" colab_type="text"
# # Introduction
#
# First, we'll go over a regular `LightningModule` implementation without the use of a `LightningDataModule`

# + id="eg-xDlmDdAwy" colab_type="code" colab={}
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms


# + [markdown] id="DzgY7wi88UuG" colab_type="text"
# ## Defining the LitMNISTModel
#
# Below, we reuse a `LightningModule` from our hello world tutorial that classifies MNIST Handwritten Digits.
#
# Unfortunately, we have hardcoded dataset-specific items within the model, forever limiting it to working with MNIST Data. 😢
#
# This is fine if you don't plan on training/evaluating your model on different datasets. However, in many cases, this can become bothersome when you want to try out your architecture with different datasets.

# + id="IQkW8_FF5nU2" colab_type="code" colab={}
class LitMNIST(pl.LightningModule):
    
    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # We hardcode dataset specific stuff here.
        self.data_dir = data_dir
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Build model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


# + [markdown] id="K7sg9KQd-QIO" colab_type="text"
# ## Training the ListMNIST Model

# + id="QxDNDaus6byD" colab_type="code" colab={}
model = LitMNIST()
trainer = pl.Trainer(max_epochs=2, gpus=1, progress_bar_refresh_rate=20)
trainer.fit(model)


# + [markdown] id="dY8d6GxmB0YU" colab_type="text"
# # Using DataModules
#
# DataModules are a way of decoupling data-related hooks from the `LightningModule` so you can develop dataset agnostic models.

# + [markdown] id="eJeT5bW081wn" colab_type="text"
# ## Defining The MNISTDataModule
#
# Let's go over each function in the class below and talk about what they're doing:
#
# 1. ```__init__```
#     - Takes in a `data_dir` arg that points to where you have downloaded/wish to download the MNIST dataset.
#     - Defines a transform that will be applied across train, val, and test dataset splits.
#     - Defines default `self.dims`, which is a tuple returned from `datamodule.size()` that can help you initialize models.
#
#
# 2. ```prepare_data```
#     - This is where we can download the dataset. We point to our desired dataset and ask torchvision's `MNIST` dataset class to download if the dataset isn't found there.
#     - **Note we do not make any state assignments in this function** (i.e. `self.something = ...`)
#
# 3. ```setup```
#     - Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test). 
#     - Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.
#     - If you don't mind loading all your datasets at once, you can set up a condition to allow for both 'fit' related setup and 'test' related setup to run whenever `None` is passed to `stage`.
#     - **Note this runs across all GPUs and it *is* safe to make state assignments here**
#
#
# 4. ```x_dataloader```
#     - `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` all return PyTorch `DataLoader` instances that are created by wrapping their respective datasets that we prepared in `setup()`

# + id="DfGKyGwG_X9v" colab_type="code" colab={}
class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


# + [markdown] id="H2Yoj-9M9dS7" colab_type="text"
# ## Defining the dataset agnostic `LitModel`
#
# Below, we define the same model as the `LitMNIST` model we made earlier. 
#
# However, this time our model has the freedom to use any input data that we'd like 🔥.

# + id="PM2IISuOBDIu" colab_type="code" colab={}
class LitModel(pl.LightningModule):
    
    def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# + [markdown] id="G4Z5olPe-xEo" colab_type="text"
# ## Training the `LitModel` using the `MNISTDataModule`
#
# Now, we initialize and train the `LitModel` using the `MNISTDataModule`'s configuration settings and dataloaders.

# + id="kV48vP_9mEli" colab_type="code" colab={}
# Init DataModule
dm = MNISTDataModule()
# Init model from datamodule's attributes
model = LitModel(*dm.size(), dm.num_classes)
# Init trainer
trainer = pl.Trainer(max_epochs=3, progress_bar_refresh_rate=20, gpus=1)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model, dm)


# + [markdown] id="WNxrugIGRRv5" colab_type="text"
# ## Defining the CIFAR10 DataModule
#
# Lets prove the `LitModel` we made earlier is dataset agnostic by defining a new datamodule for the CIFAR10 dataset.

# + id="1tkaYLU7RT5P" colab_type="code" colab={}
class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=32)


# + [markdown] id="BrXxf3oX_gsZ" colab_type="text"
# ## Training the `LitModel` using the `CIFAR10DataModule`
#
# Our model isn't very good, so it will perform pretty badly on the CIFAR10 dataset.
#
# The point here is that we can see that our `LitModel` has no problem using a different datamodule as its input data.

# + id="sd-SbWi_krdj" colab_type="code" colab={}
dm = CIFAR10DataModule()
model = LitModel(*dm.size(), dm.num_classes, hidden_size=256)
trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=20, gpus=1)
trainer.fit(model, dm)
