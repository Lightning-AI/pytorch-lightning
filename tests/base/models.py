from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from tests.base import EvalModelTemplate
from tests.base.datasets import TrialMNIST

try:
    from test_tube import HyperOptArgumentParser
except ImportError:
    # TODO: this should be discussed and moved out of this package
    raise ImportError('Missing test-tube package.')

from pytorch_lightning.core.lightning import LightningModule


class DictHparamsModel(LightningModule):

    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams
        self.l1 = torch.nn.Linear(hparams.get('in_features'), hparams['out_features'])

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(TrialMNIST(train=True, download=True), batch_size=16)


class TestModelBase(LightningModule):
    """Base LightningModule for testing. Implements only the required interface."""

    def __init__(self, hparams, force_remove_distributed_sampler: bool = False):
        """Pass in parsed HyperOptArgumentParser to the model."""
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 28 * 28)

        # remove to test warning for dist sampler
        self.force_remove_distributed_sampler = force_remove_distributed_sampler

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """Layout model."""
        self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """No special modification required for lightning, define as you normally would."""
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        # alternate possible outputs to test
        if self.trainer.batch_idx % 1 == 0:
            output = OrderedDict({
                'loss': loss_val,
                'progress_bar': {'some_val': loss_val * loss_val},
                'log': {'train_some_val': loss_val * loss_val},
            })

            return output
        if self.trainer.batch_idx % 2 == 0:
            return loss_val

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        # try no scheduler for this model (testing purposes)
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        _ = TrialMNIST(root=self.hparams.data_root, train=True, download=True)

    def _dataloader(self, train):
        # init data generators
        dataset = TrialMNIST(root=self.hparams.data_root, train=train, download=True)

        # when using multi-node we need to add the datasampler
        batch_size = self.hparams.batch_size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train
        )

        return loader


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
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
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
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
        validity = self.model(img_flat)

        return validity


class TestGAN(LightningModule):
    """Implements a basic GAN for the purpose of illustrating multiple optimizers."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=hparams.hidden_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.hidden_dim)
            z = z.type_as(imgs)

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(fake)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return DataLoader(TrialMNIST(train=True, download=True), batch_size=16)
