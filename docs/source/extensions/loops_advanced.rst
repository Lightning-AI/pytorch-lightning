:orphan:

Loops (Advanced)
================


Example: The Training Step as a Generator
---------------------------------------------------

Lightning supports multiple optimizers and offers a special :code:`training_step` flavor for it, where an extra argument with the current optimizer being used is passed in.
Take as an example the following training step of a DCGAN from the `Lightning Bolts <https://github.com/PyTorchLightning/lightning-bolts/>`_ repository:

.. code-block:: python

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # this gets computed in both cases:
        fake = self.generator(noise)

        # train discriminator
        if optimizer_idx == 0:
            # train with real
            real_pred = self.discriminator(real)
            real_loss = self.criterion(real_pred, ...)

            # train with fake
            fake_pred = self.discriminator(fake)
            fake_loss = self.criterion(fake_pred, ...)
            return real_loss + fake_loss

        # train generator
        if optimizer_idx == 1:
            fake_pred = self.discriminator(fake)
            gen_loss = self.criterion(fake_pred, fake_gt)
            return gen_loss


We notice here that the same generator `fake` outputs are needed in both optimizer cases, but if we wanted to share that computation between the two optimization steps for efficiency, there would be no elegant way to do so.
However, if we could :code:`yield` from the training step instead of returning, we can retain the local variables across training_step boundaries when we switch from one optimizer to the next in a natural way.
But such a mechanism does not exist in Lightning, therefore we need to build a custom loop for it!


.. code-block:: python

    from functools import partial
    from pytorch_lightning.loops import Loop, OptimizerLoop
    from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
    from pytorch_lightning.loops.utilities import _build_training_step_kwargs


    class YieldLoop(OptimizerLoop):
        def __init__(self):
            super().__init__()
            self._generator = None

        def on_run_start(self, batch, optimizers, batch_idx):
            super().on_run_start(batch, optimizers, batch_idx)
            assert self.trainer.lightning_module.automatic_optimization

            # We request the generator once and save it for later
            # so we can call next() on it.
            self._generator = self._get_generator(batch, batch_idx, opt_idx=0)

        def _get_generator(self, batch, batch_idx, opt_idx):
            step_kwargs = _build_training_step_kwargs(
                self.trainer.lightning_module,
                self.trainer.optimizers,
                batch,
                batch_idx,
                opt_idx,
                hiddens=None,
            )

            # Here we are basically calling lightning_module.training_step()
            # and this returns a generator! The training_step is handled by the
            # accelerator to enable distributed training.
            generator = self.trainer.accelerator.training_step(step_kwargs)
            return generator

        def _make_step_fn(self, batch, batch_idx, opt_idx):
            return partial(self._training_step, self._generator)

        def _training_step(self, generator):
            lightning_module = self.trainer.lightning_module

            # Here, instead of calling lightning_module.training_step()
            # we call next() on the generator!
            training_step_output = next(generator)

            self.trainer.accelerator.post_training_step()
            training_step_output = self.trainer.call_hook(
                "training_step_end",
                training_step_output,
            )
            result = ClosureResult.from_training_step_output(
                training_step_output,
                self.trainer.accumulate_grad_batches,
            )
            return result


Here we subclass the existing :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` and modify the way it interacts with the model's :code:`training_step`.
The new loop is called :code:`YieldLoop` and contains a reference to the generator returned by the :code:`training_step`.
On every new run (over the optimizers) we call the :code:`training_step` method on the LightningModule which is supposed to return a generator as it contains the :code:`yield` statements.
There must be as many :code:`yield` statements as there are optimizers.

The alternative to this example *manual optimization* where the same can be achieved, but with the generator loop we can still get all benefits of manual optimization without having to call backward or zero grad ourselves.

Given this new loop definition, here is how you connect it to the :code:`Trainer`:

.. code-block:: python

    model = LitModel()
    trainer = Trainer()

    yield_loop = YieldLoop()

    # the batch loop owns the optimizer loop
    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=yield_loop)

    # run the new loop!
    trainer.fit(model)

Finally, we can rewrite the GAN training step using the new yield mechanism:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        real, _ = batch

        # this gets computed only once!
        fake = self.generator(noise)

        # train discriminator, then yield
        real_pred = self.discriminator(real)
        real_loss = self.criterion(real_pred, ...)
        fake_pred = self.discriminator(fake)
        fake_loss = self.criterion(fake_pred, ...)
        yield real_loss + fake_loss

        # train generator, then yield
        fake_pred = self.discriminator(fake)
        gen_loss = self.criterion(fake_pred, fake_gt)
        yield gen_loss

Persisting the state of loops
-----------------------------

.. note::

    This is an experimental feature and is not activated by default.
    Set the environment variable `PL_FAULT_TOLERANT_TRAINING = 1` to enable saving the progress of loops.
    Read more about :doc:`fault-tolerant training training <../advanced/fault_tolerant_training>`.

A powerful property of the class-based loop interface is that it can model state.
Loop instances can save their state to the checkpoint through corresponding hooks and if implemented accordingly, resume the state of exectuion at the appropriate place.
This design is particularly interesting for fault-tolerant training which is an experimental feature released in Lightning v1.5.

The two hooks :class:`~pytorch_lightning.loops.base.Loop.on_save_checkpoint` and :class:`~pytorch_lightning.loops.base.Loop.on_load_checkpoint` function very similarly to how LightningModules and Callbacks save and load state.

.. code-block:: python

    def on_save_checkpoint(self):
        state_dict["iteration"] = self.iteration
        return state_dict


    def on_load_checkpoint(self, state_dict):
        self.iteration = state_dict["iteration"]

When the Trainer is restaring from a checkpoint (e.g., through :code:`Trainer(resume_from_checkpoint=...)`), the loop exposes a boolean :attr:`~pytorch_lightning.loops.base.Loop.restarting`.
Based around the value of this variable, the user can write the loop in such a way that it can restart from an arbitrary point given the state loaded from the checkpoint.
For example, the implementation of the :meth:`~pytorch_lightning.loops.base.Loop.reset` method could look like this given our previous example:

.. code-block:: python

    def reset(self):
        if not self.restarting:
            self.iteration = 0
