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
import operator

import torch
import torch.nn.functional as F
from lightning_utilities.core.imports import compare_version
from torch import nn
from torchmetrics import Accuracy, MeanSquaredError

from lightning.pytorch import LightningModule
from .advanced_models import Generator, Discriminator

# using new API with task
_TM_GE_0_11 = compare_version("torchmetrics", operator.ge, "0.11.0")


class ClassificationModel(LightningModule):
    def __init__(self, num_features=32, num_classes=3, lr=0.01):
        super().__init__()

        self.lr = lr
        for i in range(3):
            setattr(self, f"layer_{i}", nn.Linear(num_features, num_features))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(num_features, 3))

        acc = Accuracy(task="multiclass", num_classes=num_classes) if _TM_GE_0_11 else Accuracy()
        self.train_acc = acc.clone()
        self.valid_acc = acc.clone()
        self.test_acc = acc.clone()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)


class RegressionModel(LightningModule):
    def __init__(self):
        super().__init__()
        setattr(self, "layer_0", nn.Linear(16, 64))
        setattr(self, "layer_0a", torch.nn.ReLU())
        for i in range(1, 3):
            setattr(self, f"layer_{i}", nn.Linear(64, 64))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(64, 1))

        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        return self.layer_end(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_MSE", self.train_mse(out, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.log("val_loss", F.mse_loss(out, y), prog_bar=False)
        self.log("val_MSE", self.valid_mse(out, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.log("test_loss", F.mse_loss(out, y), prog_bar=False)
        self.log("test_MSE", self.test_mse(out, y), prog_bar=True)


class GenerationModel(LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.001, b1: float = 0.5, b2: float = 0.999):
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

        self.example_input_array = torch.rand(2, self.hidden_dim)

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        lr = self.learning_rate
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        optimizer1, optimizer2 = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hidden_dim, device=imgs.device, dtype=imgs.dtype)

        # train generator
        self.toggle_optimizer(optimizer1)
        self.generated_imgs = self.generator(z)

        valid = torch.ones(imgs.shape[0], 1, device=imgs.device, dtype=imgs.dtype)
        fake = torch.zeros(imgs.shape[0], 1, device=imgs.device, dtype=imgs.dtype)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.manual_backward(g_loss)
        optimizer1.step()
        optimizer1.zero_grad()
        self.untoggle_optimizer(optimizer1)

        # train discriminator
        self.toggle_optimizer(optimizer2)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        optimizer2.step()
        optimizer2.zero_grad()
        self.untoggle_optimizer(optimizer2)

        self.log("train/g_loss", g_loss, prog_bar=True, logger=True)
        self.log("train/d_loss", d_loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.type_as(next(self.generator.parameters()))

        with torch.inference_mode():
            z = torch.randn(imgs.shape[0], self.hidden_dim, device=imgs.device, dtype=imgs.dtype)

            fake_imgs = self(z)

            valid = torch.ones(imgs.shape[0], 1, device=imgs.device, dtype=imgs.dtype)
            fake = torch.zeros(imgs.shape[0], 1, device=imgs.device, dtype=imgs.dtype)

            g_loss = self.adversarial_loss(self.discriminator(fake_imgs), valid)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
            fake_loss = self.adversarial_loss(self.discriminator(fake_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2

        self.log("valid/g_loss", g_loss)
        self.log("valid/d_loss", d_loss)

        return {
            "valid/g_loss": g_loss.detach(),
            "valid/d_loss": d_loss.detach(),
        }

    def test_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.type_as(next(self.generator.parameters()))

        # fix for reproducibility
        g = torch.Generator(device=imgs.device)
        g.manual_seed(1234 + batch_idx)

        with torch.inference_mode():
            z = torch.randn(imgs.shape[0], self.hidden_dim, generator=g, device=imgs.device, dtype=imgs.dtype)
            fake_imgs = self(z)

            valid = torch.ones(imgs.shape[0], 1, device=imgs.device, dtype=imgs.dtype)
            fake = torch.zeros(imgs.shape[0], 1, device=imgs.device, dtype=imgs.dtype)

            g_loss = self.adversarial_loss(self.discriminator(fake_imgs), valid)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
            fake_loss = self.adversarial_loss(self.discriminator(fake_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2

        self.log("test/g_loss", g_loss)
        self.log("test/d_loss", d_loss)

        return {
            "test/g_loss": g_loss.detach(),
            "test/d_loss": d_loss.detach(),
        }
