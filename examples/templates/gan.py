"""
To run this template just do:  
python gan.py   

After a few epochs, launch tensorboard to see the images being generated at every batch.   

tensorboard --logdir default
"""
from argparse import ArgumentParser
import os
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl
from test_tube import Experiment


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
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
        super(Discriminator, self).__init__()

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


class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=hparams.latent_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        # cache for generated images
        self.generated_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_i):
        imgs, _ = batch

        # train generator
        if optimizer_i == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                z = z.cuda(imgs.device.index)

            # generate images
            self.generated_imgs = self.forward(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)

            return g_loss

        # train discriminator
        if optimizer_i == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)


def main(hparams):
    # save tensorboard logs
    exp = Experiment(save_dir=os.getcwd())

    # init model
    model = GAN(hparams)

    # fit trainer on CPU
    trainer = pl.Trainer(experiment=exp, max_nb_epochs=200)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)
