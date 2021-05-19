.. _optimizers:

************
Optimization
************
Lightning offers two modes for managing the optimization process:

- automatic optimization
- manual optimization

For the majority of research cases, **automatic optimization** will do the right thing for you and it is what most
users should use.

For advanced/expert users who want to do esoteric optimization schedules or techniques, use **manual optimization**.

-----

.. _manual_optimization:

Manual optimization
===================
For advanced research topics like reinforcement learning, sparse coding, or GAN research, it may be desirable to
manually manage the optimization process.

This is only recommended for experts who need ultimate flexibility.
Lightning will handle only precision and accelerators logic.
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

-----

Gradient accumulation
---------------------
You can accumulate gradients over batches similarly to
:attr:`~pytorch_lightning.trainer.Trainer.accumulate_grad_batches` of automatic optimization.
To perform gradient accumulation with one optimizer, you can do as such.

.. testcode:: python

    # accumulate gradients over `n` batches
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss = self.compute_loss(batch)
        self.manual_backward(loss)

        # accumulate gradients of `n` batches
        if (batch_idx + 1) % n == 0:
            opt.step()
            opt.zero_grad()

-----

Use multiple optimizers (like GANs) [manual]
--------------------------------------------
Here is an example training a simple GAN with multiple optimizers.

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

            errD = (errD_real + errD_fake)

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

            self.log_dict({'g_loss': errG, 'd_loss': errD}, prog_bar=True)

        def configure_optimizers(self):
            g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
            d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
            return g_opt, d_opt

-----

Learning rate scheduling
------------------------
Every optimizer you use can be paired with any
`Learning Rate Scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_. Please see the
documentation of :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers` for all the available options

-----

Learning rate scheduling [manual]
---------------------------------
You can call ``lr_scheduler.step()`` at arbitrary intervals.
Use ``self.lr_schedulers()`` in  your :class:`~pytorch_lightning.core.lightning.LightningModule` to access any learning rate schedulers
defined in your :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`.

.. warning::
   * Before 1.3, Lightning automatically called ``lr_scheduler.step()`` in both automatic and manual optimization. From
     1.3, ``lr_scheduler.step()`` is now for the user to call at arbitrary intervals.
   * Note that the ``lr_dict`` keys, such as ``"step"`` and ``""interval"``, will be ignored even if they are provided in
     your :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers` during manual optimization.

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

If you want to call ``lr_scheduler.step()`` every ``n`` steps/epochs, do the following.

.. testcode:: python

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # do forward, backward, and optimization
        ...

        sch = self.lr_schedulers()

        # step every `n` batches
        if (batch_idx + 1) % n == 0:
            sch.step()

        # step every `n` epochs
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % n == 0:
            sch.step()

-----

Improve training speed with model toggling
------------------------------------------
Toggling models can improve your training speed when performing gradient accumulation with multiple optimizers in a
distributed setting.

Here is an explanation of what it does:

* Considering the current optimizer as A and all other optimizers as B.
* Toggling means that all parameters from B exclusive to A will have their ``requires_grad`` attribute set to ``False``.
* Their original state will be restored when exiting the context manager.

When performing gradient accumulation, there is no need to perform grad synchronization during the accumulation phase.
Setting ``sync_grad`` to ``False`` will block this synchronization and improve your training speed.

:class:`~pytorch_lightning.core.optimizer.LightningOptimizer` provides a
:meth:`~pytorch_lightning.core.optimizer.LightningOptimizer.toggle_model` function as a
:func:`contextlib.contextmanager` for advanced users.

Here is an example for advanced use-case.

.. testcode:: python

    # Scenario for a GAN with gradient accumulation every 2 batches and optimized for multiple gpus.
    class SimpleGAN(LightningModule):

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            # Implementation follows the PyTorch tutorial:
            # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            g_opt, d_opt = self.optimizers()

            X, _ = batch
            X.requires_grad = True
            batch_size = X.shape[0]

            real_label = torch.ones((batch_size, 1), device=self.device)
            fake_label = torch.zeros((batch_size, 1), device=self.device)

            # Sync and clear gradients
            # at the end of accumulation or
            # at the end of an epoch.
            is_last_batch_to_accumulate = \
                (batch_idx + 1) % 2 == 0 or self.trainer.is_last_batch

            g_X = self.sample_G(batch_size)

            ##########################
            # Optimize Discriminator #
            ##########################
            with d_opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
                d_x = self.D(X)
                errD_real = self.criterion(d_x, real_label)

                d_z = self.D(g_X.detach())
                errD_fake = self.criterion(d_z, fake_label)

                errD = (errD_real + errD_fake)

                self.manual_backward(errD)
                if is_last_batch_to_accumulate:
                    d_opt.step()
                    d_opt.zero_grad()

            ######################
            # Optimize Generator #
            ######################
            with g_opt.toggle_model(sync_grad=is_last_batch_to_accumulate):
                d_z = self.D(g_X)
                errG = self.criterion(d_z, real_label)

                self.manual_backward(errG)
                if is_last_batch_to_accumulate:
                    g_opt.step()
                    g_opt.zero_grad()

            self.log_dict({'g_loss': errG, 'd_loss': errD}, prog_bar=True)

-----

Use closure for LBFGS-like optimizers
-------------------------------------
It is a good practice to provide the optimizer with a closure function that performs a ``forward``, ``zero_grad`` and
``backward`` of your model. It is optional for most optimizers, but makes your code compatible if you switch to an
optimizer which requires a closure, such as :class:`torch.optim.LBFGS`.

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

------

Access your own optimizer [manual]
----------------------------------
``optimizer`` is a :class:`~pytorch_lightning.core.optimizer.LightningOptimizer` object wrapping your own optimizer
configured in your :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`. You can access your own optimizer
with ``optimizer.optimizer``. However, if you use your own optimizer to perform a step, Lightning won't be able to
support accelerators and precision for you.

.. testcode:: python

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(batch, batch_idx):
        optimizer = self.optimizers()

        # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
        # To access it, do the following.
        # However, it won't work on TPU, AMP, etc...
        optimizer = optimizer.optimizer
        ...

-----

Automatic optimization
======================
With Lightning, most users don't have to think about when to call ``.zero_grad()``, ``.backward()`` and ``.step()``
since Lightning automates that for you.

Under the hood, Lightning does the following:

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            loss = model.training_step(batch, batch_idx, ...)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

In the case of multiple optimizers, Lightning does the following:

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            for opt in optimizers:
                loss = model.training_step(batch, batch_idx, optimizer_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

.. warning::
   Before 1.2.2, Lightning internally calls ``backward``, ``step`` and ``zero_grad`` in the order.
   From 1.2.2, the order is changed to ``zero_grad``, ``backward`` and ``step``.

-----

Use multiple optimizers (like GANs)
-----------------------------------
To use multiple optimizers (optionally with learning rate schedulers), return two or more optimizers from
:meth:`~pytorch_lightning.core.LightningModule.configure_optimizers`.

.. testcode:: python

    # two optimizers, no schedulers
    def configure_optimizers(self):
        return Adam(...), SGD(...)

    # two optimizers, one scheduler for adam only
    def configure_optimizers(self):
        opt1 = Adam(...)
        opt2 = SGD(...)
        optimizers = [opt1, opt2]
        lr_schedulers = {'scheduler': ReduceLROnPlateau(opt1, ...), 'monitor': 'metric_to_track'}
        return optimizers, lr_schedulers

    # two optimizers, two schedulers
    def configure_optimizers(self):
        opt1 = Adam(...)
        opt2 = SGD(...)
        return [opt1, opt2], [StepLR(opt1, ...), OneCycleLR(opt2, ...)]

Under the hood, Lightning will call each optimizer sequentially:

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            for opt in optimizers:
                loss = train_step(batch, batch_idx, optimizer_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

-----

Step optimizers at arbitrary intervals
--------------------------------------
To do more interesting things with your optimizers such as learning rate warm-up or odd scheduling,
override the :meth:`~pytorch_lightning.core.lightning.LightningModule.optimizer_step` function.

.. warning::
    If you are overriding this method, make sure that you pass the ``optimizer_closure`` parameter to
    ``optimizer.step()`` function as shown in the examples because ``training_step()``, ``optimizer.zero_grad()``,
    ``backward()`` are called in the closure function.

For example, here step optimizer A every batch and optimizer B every 2 batches.

.. testcode:: python

    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        # update generator every step
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        # update discriminator every 2 steps
        if optimizer_idx == 1:
            if (batch_idx + 1) % 2 == 0:
                optimizer.step(closure=optimizer_closure)

        # ...
        # add as many optimizers as you want

Here we add a learning rate warm-up.

.. testcode:: python

    # learning rate warm-up
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        # skip the first 500 steps
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step(closure=optimizer_closure)

-----

Access your own optimizer
-------------------------
``optimizer`` is a :class:`~pytorch_lightning.core.optimizer.LightningOptimizer` object wrapping your own optimizer
configured in your :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`.
You can access your own optimizer with ``optimizer.optimizer``. However, if you use your own optimizer
to perform a step, Lightning won't be able to support accelerators and precision for you.

.. testcode:: python

    # function hook in LightningModule
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)

    # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
    # To access it, do the following.
    # However, it won't work on TPU, AMP, etc...
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        optimizer = optimizer.optimizer
        optimizer.step(closure=optimizer_closure)
