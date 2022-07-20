# Copyright The PyTorch Lightning team.
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
import inspect
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision

#############################################################################################
#                                    Yield Loop                                             #
#                                                                                           #
# This example shows an implementation of a custom loop that changes how the                #
# `LightningModule.training_step` behaves. In particular, this custom "Yield" loop will     #
# enable the `training_step` to yield like a Python generator, retaining the values         #
# of local variables for subsequent calls. This can result in much cleaner and elegant      #
# code when dealing with multiple optimizers (automatic optimization).                      #
#                                                                                           #
# Learn more about the loop structure from the documentation:                               #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################


#############################################################################################
#                        Step 1 / 3: Implement a custom OptimizerLoop                       #
#                                                                                           #
# The `training_step` gets called in the                                                    #
# `pytorch_lightning.loops.optimization.OptimizerLoop`. To make it into a Python generator, #
# we need to override the place where it gets called.                                       #
#############################################################################################


class YieldLoop(OptimizerLoop):
    def __init__(self):
        super().__init__()
        self._generator = None

    def connect(self, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def on_run_start(self, optimizers, kwargs):
        super().on_run_start(optimizers, kwargs)
        if not inspect.isgeneratorfunction(self.trainer.lightning_module.training_step):
            raise MisconfigurationException("The `LightningModule` does not yield anything in the `training_step`.")
        assert self.trainer.lightning_module.automatic_optimization

        # We request the generator once and save it for later so we can call next() on it.
        self._generator = self._get_generator(kwargs)

    def _make_step_fn(self, *_):
        return partial(self._training_step, self._generator)

    def _get_generator(self, kwargs, opt_idx=0):
        kwargs = self._build_kwargs(kwargs, opt_idx, hiddens=None)

        # Here we are basically calling `lightning_module.training_step()`
        # and this returns a generator! The `training_step` is handled by
        # the accelerator to enable distributed training.
        return self.trainer.strategy.training_step(*kwargs.values())

    def _training_step(self, generator):
        # required for logging
        self.trainer.lightning_module._current_fx_name = "training_step"

        # Here, instead of calling `lightning_module.training_step()`
        # we call next() on the generator!
        training_step_output = next(generator)
        self.trainer.strategy.post_training_step()

        model_output = self.trainer._call_lightning_module_hook("training_step_end", training_step_output)
        strategy_output = self.trainer._call_strategy_hook("training_step_end", training_step_output)
        training_step_output = strategy_output if model_output is None else model_output

        # The closure result takes care of properly detaching the loss for logging and peforms
        # some additional checks that the output format is correct.
        result = ClosureResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)
        return result


#############################################################################################
#               Step 2 / 3: Implement a model using the new yield mechanism                 #
#                                                                                           #
# We can now implement a model that defines the `training_step` using "yield" statements.   #
# We choose a generative adversarial network (GAN) because it alternates between two        #
# optimizers updating the model parameters. In the first step we compute the loss of the    #
# first network (coincidentally also named "generator") and yield the loss. In the second   #
# step we compute the loss of the second network (the "discriminator") and yield again.     #
# The nice property of this yield approach is that we can reuse variables that we computed  #
# earlier. If this was a regular Lightning `training_step`, we would have to recompute the  #
# output of the first network.                                                              #
#############################################################################################


class Generator(nn.Module):
    """
    >>> Generator(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Generator(
      (model): Sequential(...)
    )
    """

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
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
    """
    >>> Discriminator(img_shape=(1, 28, 28))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Discriminator(
      (model): Sequential(...)
    )
    """

    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(LightningModule):
    """
    >>> GAN(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GAN(
      (generator): Generator(
        (model): Sequential(...)
      )
      (discriminator): Discriminator(
        (model): Sequential(...)
      )
    )
    """

    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
    ):
        super().__init__()

        self.save_hyperparameters()

        # networks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("GAN")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        return parser_out

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    # This training_step method is now a Python generator
    def training_step(self, batch, batch_idx, optimizer_idx=0) -> Generator:
        imgs, _ = batch
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Here, we compute the generator output once and reuse it later.
        # It gets saved when we yield from the training_step.
        # The output then gets re-used again in the discriminator update.
        generator_output = self(z)

        # train generator
        real_labels = torch.ones(imgs.size(0), 1)
        real_labels = real_labels.type_as(imgs)
        g_loss = self.adversarial_loss(self.discriminator(generator_output), real_labels)
        self.log("g_loss", g_loss)

        # Yield instead of return: This makes the training_step a Python generator.
        # Once we call it again, it will continue the execution with the block below
        yield g_loss

        # train discriminator
        real_labels = torch.ones(imgs.size(0), 1)
        real_labels = real_labels.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs), real_labels)
        fake_labels = torch.zeros(imgs.size(0), 1)
        fake_labels = fake_labels.type_as(imgs)

        # We make use again of the generator_output
        fake_loss = self.adversarial_loss(self.discriminator(generator_output.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss)

        yield d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            logger.experiment.add_image("generated_images", grid, self.current_epoch)


#############################################################################################
#                       Step 3 / 3: Connect the loop to the Trainer                         #
#                                                                                           #
# Finally, attach the loop to the `Trainer`. Here, we modified the `AutomaticOptimization`  #
# loop which is a subloop of the `TrainingBatchLoop`. We use `.connect()` to attach it.     #
#############################################################################################

if __name__ == "__main__":
    model = GAN()
    dm = MNISTDataModule()
    trainer = Trainer()

    # Connect the new loop
    # YieldLoop now replaces the previous optimizer loop
    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=YieldLoop())

    # fit() will now use the new loop!
    trainer.fit(model, dm)
