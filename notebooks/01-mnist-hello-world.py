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
# <a href="https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="i7XbLCXGkll9" colab_type="text"
# # Introduction to Pytorch Lightning ⚡
#
# In this notebook, we'll go over the basics of lightning by preparing models to train on the [MNIST Handwritten Digits dataset](https://en.wikipedia.org/wiki/MNIST_database).
#
# ---
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)

# + [markdown] id="2LODD6w9ixlT" colab_type="text"
# ### Setup  
# Lightning is easy to install. Simply ```pip install pytorch-lightning```

# + id="zK7-Gg69kMnG" colab_type="code" colab={}
# ! pip install pytorch-lightning --quiet

# + id="w4_TYnt_keJi" colab_type="code" colab={}
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


# + [markdown] id="EHpyMPKFkVbZ" colab_type="text"
# ## Simplest example
#
# Here's the simplest most minimal example with just a training loop (no validation, no testing).
#
# **Keep in Mind** - A `LightningModule` *is* a PyTorch `nn.Module` - it just has a few more helpful features.

# + id="V7ELesz1kVQo" colab_type="code" colab={}
class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return pl.TrainResult(loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# + [markdown] id="hIrtHg-Dv8TJ" colab_type="text"
# By using the `Trainer` you automatically get:
# 1. Tensorboard logging
# 2. Model checkpointing
# 3. Training and validation loop
# 4. early-stopping

# + id="4Dk6Ykv8lI7X" colab_type="code" colab={}
# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

# Initialize a trainer
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)


# + [markdown] id="KNpOoBeIjscS" colab_type="text"
# ## A more complete MNIST Lightning Module Example
#
# That wasn't so hard was it?
#
# Now that we've got our feet wet, let's dive in a bit deeper and write a more complete `LightningModule` for MNIST...
#
# This time, we'll bake in all the dataset specific pieces directly in the `LightningModule`. This way, we can avoid writing extra code at the beginning of our script every time we want to run it.
#
# ---
#
# ### Note what the following built-in functions are doing:
#
# 1. [prepare_data()](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.prepare_data) 💾
#     - This is where we can download the dataset. We point to our desired dataset and ask torchvision's `MNIST` dataset class to download if the dataset isn't found there.
#     - **Note we do not make any state assignments in this function** (i.e. `self.something = ...`)
#
# 2. [setup(stage)](https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html#setup) ⚙️
#     - Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test). 
#     - Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.
#     - If you don't mind loading all your datasets at once, you can set up a condition to allow for both 'fit' related setup and 'test' related setup to run whenever `None` is passed to `stage` (or ignore it altogether and exclude any conditionals).
#     - **Note this runs across all GPUs and it *is* safe to make state assignments here**
#
# 3. [x_dataloader()](https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html#data-hooks) ♻️
#     - `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` all return PyTorch `DataLoader` instances that are created by wrapping their respective datasets that we prepared in `setup()`

# + id="4DNItffri95Q" colab_type="code" colab={}
class LitMNIST(pl.LightningModule):
    
    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define PyTorch model
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

        # Calling result.log will surface up scalars for you in TensorBoard
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

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


# + id="Mb0U5Rk2kLBy" colab_type="code" colab={}
model = LitMNIST()
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)
trainer.fit(model)

# + [markdown] id="nht8AvMptY6I" colab_type="text"
# ### Testing
#
# To test a model, call `trainer.test(model)`.
#
# Or, if you've just trained a model, you can just call `trainer.test()` and Lightning will automatically test using the best saved checkpoint (conditioned on val_loss).

# + id="PA151FkLtprO" colab_type="code" colab={}
trainer.test()

# + [markdown] id="T3-3lbbNtr5T" colab_type="text"
# ### Bonus Tip
#
# You can keep calling `trainer.fit(model)` as many times as you'd like to continue training

# + id="IFBwCbLet2r6" colab_type="code" colab={}
trainer.fit(model)

# + [markdown] id="8TRyS5CCt3n9" colab_type="text"
# In Colab, you can use the TensorBoard magic function to view the logs that Lightning has created for you!

# + id="wizS-QiLuAYo" colab_type="code" colab={}
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
