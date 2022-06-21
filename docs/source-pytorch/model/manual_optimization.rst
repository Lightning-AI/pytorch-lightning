*******************
Manual Optimization
*******************

For advanced research topics like reinforcement learning, sparse coding, or GAN research, it may be desirable to
manually manage the optimization process.

This is only recommended for experts who need ultimate flexibility.
Lightning will handle only accelerator, precision and strategy logic.
The users are left with ``optimizer.zero_grad()``, gradient accumulation, model toggling, etc..

To manually optimize, do the following:

* Set ``self.automatic_optimization=False`` in your ``LightningModule``'s ``__init__``.
* Use the following functions and call them manually:

  * ``self.optimizers()`` to access your optimizers (one or multiple)
  * ``optimizer.zero_grad()`` to clear the gradients from the previous training step
  * ``self.manual_backward(loss)`` instead of ``loss.backward()``
  * ``optimizer.step()`` to update your model parameters

Here is a minimal example of manual optimization.

.. testcode:: python

    from pytorch_lightning import LightningModule


    class MyModel(LightningModule):
        def __init__(self):
            super().__init__()
            # Important: This property activates manual optimization.
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            opt = self.optimizers()
            opt.zero_grad()
            loss = self.compute_loss(batch)
            self.manual_backward(loss)
            opt.step()

.. warning::
   Before 1.2, ``optimizer.step()`` was calling ``optimizer.zero_grad()`` internally.
   From 1.2, it is left to the user's expertise.

.. tip::
   Be careful where you call ``optimizer.zero_grad()``, or your model won't converge.
   It is good practice to call ``optimizer.zero_grad()`` before ``self.manual_backward(loss)``.


Access your Own Optimizer
=========================

The provided ``optimizer`` is a :class:`~pytorch_lightning.core.optimizer.LightningOptimizer` object wrapping your own optimizer
configured in your :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`. You can access your own optimizer
with ``optimizer.optimizer``. However, if you use your own optimizer to perform a step, Lightning won't be able to
support accelerators, precision and profiling for you.

.. testcode:: python

   class Model(LightningModule):
       def __init__(self):
           super().__init__()
           self.automatic_optimization = False
           ...

       def training_step(self, batch, batch_idx):
           optimizer = self.optimizers()

           # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
           # To access it, do the following.
           # However, it won't work on TPU, AMP, etc...
           optimizer = optimizer.optimizer
           ...

Gradient Accumulation
=====================

You can accumulate gradients over batches similarly to ``accumulate_grad_batches`` argument in
:ref:`Trainer <trainer>` for automatic optimization. To perform gradient accumulation with one optimizer
after every ``N`` steps, you can do as such.

.. testcode:: python

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss = self.compute_loss(batch)
        self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()


Use Multiple Optimizers (like GANs)
===================================

Here is an example training a simple GAN with multiple optimizers using manual optimization.

.. testcode:: python

    import torch
    from torch import Tensor
    from pytorch_lightning import LightningModule


    class SimpleGAN(LightningModule):
        def __init__(self):
            super().__init__()
            self.G = Generator()
            self.D = Discriminator()

            # Important: This property activates manual optimization.
            self.automatic_optimization = False

        def sample_z(self, n) -> Tensor:
            sample = self._Z.sample((n,))
            return sample

        def sample_G(self, n) -> Tensor:
            z = self.sample_z(n)
            return self.G(z)

        def training_step(self, batch, batch_idx):
            # Implementation follows the PyTorch tutorial:
            # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            g_opt, d_opt = self.optimizers()

            X, _ = batch
            batch_size = X.shape[0]

            real_label = torch.ones((batch_size, 1), device=self.device)
            fake_label = torch.zeros((batch_size, 1), device=self.device)

            g_X = self.sample_G(batch_size)

            ##########################
            # Optimize Discriminator #
            ##########################
            d_x = self.D(X)
            errD_real = self.criterion(d_x, real_label)

            d_z = self.D(g_X.detach())
            errD_fake = self.criterion(d_z, fake_label)

            errD = errD_real + errD_fake

            d_opt.zero_grad()
            self.manual_backward(errD)
            d_opt.step()

            ######################
            # Optimize Generator #
            ######################
            d_z = self.D(g_X)
            errG = self.criterion(d_z, real_label)

            g_opt.zero_grad()
            self.manual_backward(errG)
            g_opt.step()

            self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

        def configure_optimizers(self):
            g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
            d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
            return g_opt, d_opt


Learning Rate Scheduling
========================

Every optimizer you use can be paired with any
`Learning Rate Scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_. Please see the
documentation of :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers` for all the available options

You can call ``lr_scheduler.step()`` at arbitrary intervals.
Use ``self.lr_schedulers()`` in  your :class:`~pytorch_lightning.core.module.LightningModule` to access any learning rate schedulers
defined in your :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`.

.. warning::
   * Before v1.3, Lightning automatically called ``lr_scheduler.step()`` in both automatic and manual optimization. From
     1.3, ``lr_scheduler.step()`` is now for the user to call at arbitrary intervals.
   * Note that the ``lr_scheduler_config`` keys, such as ``"frequency"`` and ``"interval"``, will be ignored even if they are provided in
     your :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers` during manual optimization.

Here is an example calling ``lr_scheduler.step()`` every step.

.. testcode:: python

    # step every batch
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        # do forward, backward, and optimization
        ...

        # single scheduler
        sch = self.lr_schedulers()
        sch.step()

        # multiple schedulers
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

If you want to call ``lr_scheduler.step()`` every ``N`` steps/epochs, do the following.

.. testcode:: python

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        # do forward, backward, and optimization
        ...

        sch = self.lr_schedulers()

        # step every N batches
        if (batch_idx + 1) % N == 0:
            sch.step()

        # step every N epochs
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % N == 0:
            sch.step()

If you want to call schedulers that require a metric value after each epoch, consider doing the following:

.. testcode::

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False


    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])

Use Closure for LBFGS-like Optimizers
=====================================

It is a good practice to provide the optimizer with a closure function that performs a ``forward``, ``zero_grad`` and
``backward`` of your model. It is optional for most optimizers, but makes your code compatible if you switch to an
optimizer which requires a closure, such as :class:`~torch.optim.LBFGS`.

See `the PyTorch docs <https://pytorch.org/docs/stable/optim.html#optimizer-step-closure>`_ for more about the closure.

Here is an example using a closure function.

.. testcode:: python

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False


    def configure_optimizers(self):
        return torch.optim.LBFGS(...)


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        def closure():
            loss = self.compute_loss(batch)
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        opt.step(closure=closure)

.. warning::
   The :class:`~torch.optim.LBFGS` optimizer is not supported for apex AMP, native AMP, IPUs, or DeepSpeed.
