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
from abc import ABC, abstractmethod
from functools import partial
from typing import Generator

import torch

from pl_examples.domain_templates.generative_adversarial_net import GAN as GANTemplate
from pl_examples.domain_templates.generative_adversarial_net import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from pytorch_lightning.loops.utilities import _build_training_step_kwargs
from pytorch_lightning.utilities.exceptions import MisconfigurationException

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
#            Step 1 / 4: Define the interface for a yielding LightningModule                #
#                                                                                           #
# Whenever you decide to redefine hooks and their signature in Lightning, it is a good      #
# practice to define a class with the expected interface.                                   #
#                                                                                           #
# Here we choose to define a new "flavor" of the `training_step`. We want to create a       #
# `training_step` from which we can "yield", like we would from a Python generator.         #
#############################################################################################


class YieldingLightningModule(ABC):
    """Interface for the LightningModule to define a flavor for automatic optimization where the training step
    method yields losses for each optimizer instead of returning them."""

    @abstractmethod
    def training_step(self, batch, batch_idx, optimizer_idx=0) -> Generator:
        pass


#############################################################################################
#                        Step 2 / 4: Implement a custom OptimizerLoop                       #
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

    def on_run_start(self, batch, optimizers, batch_idx):
        super().on_run_start(batch, optimizers, batch_idx)
        if not isinstance(self.trainer.lightning_module, YieldingLightningModule):
            raise MisconfigurationException(
                "Given LightingModule does not inherit the Yield interface for automatic optimization, but a"
                " YieldLoop was requested."
            )
        assert inspect.isgeneratorfunction(self.trainer.lightning_module.training_step)
        assert self.trainer.lightning_module.automatic_optimization

        # We request the generator once and save it for later
        # so we can call next() on it.
        self._generator = self._get_generator(batch, batch_idx, opt_idx=0)

    def _make_step_fn(self, split_batch, batch_idx, opt_idx):
        return partial(self._training_step, self._generator)

    def _get_generator(self, split_batch, batch_idx, opt_idx):
        step_kwargs = _build_training_step_kwargs(
            self.trainer.lightning_module, self.trainer.optimizers, split_batch, batch_idx, opt_idx, hiddens=None
        )

        # Here we are basically calling `lightning_module.training_step()`
        # and this returns a generator! The `training_step` is handled by the
        # accelerator to enable distributed training.
        return self.trainer.accelerator.training_step(step_kwargs)

    def _training_step(self, generator):
        # required for logging
        self.trainer.lightning_module._current_fx_name = "training_step"

        # Here, instead of calling `lightning_module.training_step()`
        # we call next() on the generator!
        training_step_output = next(generator)
        self.trainer.accelerator.post_training_step()

        training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

        # The closure result takes care of properly detaching the loss for logging and peforms
        # some additional checks that the output format is correct.
        result = ClosureResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)
        return result


#############################################################################################
#               Step 3 / 4: Implement a model using the new yield mechanism                 #
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


class GAN(YieldingLightningModule, GANTemplate):

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
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        g_loss = self.adversarial_loss(self.discriminator(generator_output), valid)
        self.log("g_loss", g_loss)

        # Yield instead of return: This makes the training_step a Python generator.
        # Once we call it again, it will continue the execution with the block below
        yield g_loss

        # train discriminator
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        # We make use again of the generator_output
        fake_loss = self.adversarial_loss(self.discriminator(generator_output.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss)

        yield d_loss


#############################################################################################
#                       Step 4 / 4: Connect the loop to the Trainer                         #
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
