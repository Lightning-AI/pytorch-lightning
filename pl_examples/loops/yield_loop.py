import inspect
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


class Yield:
    """Interface for the LightningModule to define a flavor for automatic optimization where the training step
    method yields losses for each optimizer instead of returning them."""

    def training_step(self, batch, batch_idx, optimizer_idx=0) -> Generator:
        pass


class YieldLoop(OptimizerLoop):
    def __init__(self):
        super().__init__()
        self._generator = None

    def connect(self, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def on_run_start(self, batch, optimizers, batch_idx):
        super().on_run_start(batch, optimizers, batch_idx)
        if not isinstance(self.trainer.lightning_module, Yield):
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

        # Here we are basically calling lightning_module.training_step()
        # and this returns a generator! The training_step is handled by the
        # accelerator to enable distributed training.
        return self.trainer.accelerator.training_step(step_kwargs)

    def _training_step(self, generator):
        # required for logging
        self.trainer.lightning_module._current_fx_name = "training_step"

        # Here, instead of calling lightning_module.training_step()
        # we call next() on the generator!
        training_step_output = next(generator)
        self.trainer.accelerator.post_training_step()

        training_step_output = self.trainer.call_hook("training_step_end", training_step_output)
        result = ClosureResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)
        return result


class GAN(Yield, GANTemplate):

    # This training_step method is now a generator
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        imgs, _ = batch
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Here, we compute the generator output once and reuse it later.
        # It gets saved as part of the generator
        # use it in both the generator update and the discriminator update
        generator_output = self(z)

        # train generator
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        g_loss = self.adversarial_loss(self.discriminator(generator_output), valid)
        self.log("g_loss", g_loss)

        # Yield instead of return: This makes the training_step a generator.
        # Once we call it again, it will continue the execution with the block below
        yield g_loss

        # train discriminator
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        fake_loss = self.adversarial_loss(self.discriminator(generator_output.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss)

        yield d_loss


if __name__ == "__main__":
    model = GAN()
    dm = MNISTDataModule()
    trainer = Trainer()

    # Connect the new loop
    # YieldLoop now replaces the previous optimizer loop
    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=YieldLoop())

    # fit() will now use the new loop!
    trainer.fit(model, dm)
