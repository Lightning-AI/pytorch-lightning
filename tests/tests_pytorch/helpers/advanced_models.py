# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lightning.pytorch.core.module import LightningModule
from tests_pytorch import _PATH_DATASETS
from tests_pytorch.helpers.datasets import MNIST, AverageDataset, TrialMNIST


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


class BasicGAN(LightningModule):
    """Implements a basic GAN for the purpose of illustrating multiple optimizers."""

    def __init__(
        self, hidden_dim: int = 128, learning_rate: float = 0.001, b1: float = 0.5, b2: float = 0.999, **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2

        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=self.hidden_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        self.example_input_array = torch.rand(2, self.hidden_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        self.last_imgs = imgs

        optimizer1, optimizer2 = self.optimizers()

        # train generator
        # sample noise
        self.toggle_optimizer(optimizer1)
        z = torch.randn(imgs.shape[0], self.hidden_dim)
        z = z.type_as(imgs)

        # generate images
        self.generated_imgs = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True, logger=True)
        self.manual_backward(g_loss)
        optimizer1.step()
        optimizer1.zero_grad()
        self.untoggle_optimizer(optimizer1)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer2)
        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(fake)

        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True, logger=True)
        self.manual_backward(d_loss)
        optimizer2.step()
        optimizer2.zero_grad()
        self.untoggle_optimizer(optimizer2)

    def configure_optimizers(self):
        lr = self.learning_rate
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return DataLoader(TrialMNIST(root=_PATH_DATASETS, train=True, download=True), batch_size=16)


class ParityModuleRNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, batch_first=True)
        self.linear_out = nn.Linear(in_features=20, out_features=5)
        self.example_input_array = torch.rand(2, 3, 10)
        self._loss = []  # needed for checking if the loss is the same as vanilla torch

    def forward(self, x):
        seq, _ = self.rnn(x)
        return self.linear_out(seq)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self._loss.append(loss.item())
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(AverageDataset(), batch_size=30)


class ParityModuleMNIST(LightningModule):
    def __init__(self):
        super().__init__()
        self.c_d1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.c_d1_bn = nn.BatchNorm1d(128)
        self.c_d1_drop = nn.Dropout(0.3)
        self.c_d2 = nn.Linear(in_features=128, out_features=10)
        self.example_input_array = torch.rand(2, 1, 28, 28)
        self._loss = []  # needed for checking if the loss is the same as vanilla torch

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self._loss.append(loss.item())
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(MNIST(root=_PATH_DATASETS, train=True, download=True), batch_size=128, num_workers=1)


class TBPTTModule(LightningModule):
    def __init__(self):
        super().__init__()

        self.batch_size = 10
        self.in_features = 10
        self.out_features = 5
        self.hidden_dim = 20

        self.automatic_optimization = False
        self.truncated_bptt_steps = 10

        self.rnn = nn.LSTM(self.in_features, self.hidden_dim, batch_first=True)
        self.linear_out = nn.Linear(in_features=self.hidden_dim, out_features=self.out_features)

    def forward(self, x, hs):
        seq, hs = self.rnn(x, hs)
        return self.linear_out(seq), hs

    def training_step(self, batch, batch_idx):
        x, y = batch
        split_x, split_y = [
            x.tensor_split(self.truncated_bptt_steps, dim=1),
            y.tensor_split(self.truncated_bptt_steps, dim=1),
        ]

        hiddens = None
        optimizer = self.optimizers()
        losses = []

        for x, y in zip(split_x, split_y):
            y_pred, hiddens = self(x, hiddens)
            loss = F.mse_loss(y_pred, y)

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

            # "Truncate"
            hiddens = [h.detach() for h in hiddens]
            losses.append(loss.detach())

        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(AverageDataset(), batch_size=self.batch_size)
