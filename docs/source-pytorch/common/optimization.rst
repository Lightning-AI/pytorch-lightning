:orphan:

.. _optimization:

############
Optimization
############

Lightning offers two modes for managing the optimization process:

- Manual Optimization
- Automatic Optimization

For the majority of research cases, **automatic optimization** will do the right thing for you and it is what most
users should use.

For advanced/expert users who want to do esoteric optimization schedules or techniques, use **manual optimization**.

.. _manual_optimization:

----

.. include:: ../model/manual_optimization.rst

-----

**********************
Automatic Optimization
**********************

With Lightning, most users don't have to think about when to call ``.zero_grad()``, ``.backward()`` and ``.step()``
since Lightning automates that for you.

Under the hood, Lightning does the following:

.. code-block:: python

    for epoch in epochs:
        for batch in data:

            def closure():
                loss = model.training_step(batch, batch_idx, ...)
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(closure)

        lr_scheduler.step()

In the case of multiple optimizers, Lightning does the following:

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            for opt in optimizers:

                def closure():
                    loss = model.training_step(batch, batch_idx, optimizer_idx)
                    opt.zero_grad()
                    loss.backward()
                    return loss

                opt.step(closure)

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

As can be seen in the code snippet above, Lightning defines a closure with ``training_step()``, ``optimizer.zero_grad()``
and ``loss.backward()`` for the optimization. This mechanism is in place to support optimizers which operate on the
output of the closure (e.g. the loss) or need to call the closure several times (e.g. :class:`~torch.optim.LBFGS`).

.. warning::

   Before v1.2.2, Lightning internally calls ``backward``, ``step`` and ``zero_grad`` in the order.
   From v1.2.2, the order is changed to ``zero_grad``, ``backward`` and ``step``.


Gradient Accumulation
=====================

.. include:: ../common/gradient_accumulation.rst


Use Multiple Optimizers (like GANs)
===================================

To use multiple optimizers (optionally with learning rate schedulers), return two or more optimizers from
:meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`.

.. testcode:: python

    # two optimizers, no schedulers
    def configure_optimizers(self):
        return Adam(...), SGD(...)


    # two optimizers, one scheduler for adam only
    def configure_optimizers(self):
        opt1 = Adam(...)
        opt2 = SGD(...)
        optimizers = [opt1, opt2]
        lr_schedulers = {"scheduler": ReduceLROnPlateau(opt1, ...), "monitor": "metric_to_track"}
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


Step Optimizers at Arbitrary Intervals
=======================================

To do more interesting things with your optimizers such as learning rate warm-up or odd scheduling,
override the :meth:`~pytorch_lightning.core.module.LightningModule.optimizer_step` function.

.. warning::
    If you are overriding this method, make sure that you pass the ``optimizer_closure`` parameter to
    ``optimizer.step()`` function as shown in the examples because ``training_step()``, ``optimizer.zero_grad()``,
    ``loss.backward()`` are called in the closure function.

For example, here step optimizer A every batch and optimizer B every 2 batches.

.. testcode:: python

    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update generator every step
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        # update discriminator every 2 steps
        if optimizer_idx == 1:
            if (batch_idx + 1) % 2 == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()

        # ...
        # add as many optimizers as you want

Here we add a manual learning rate warm-up without an lr scheduler.

.. testcode:: python

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate


Access your Own Optimizer
=========================

The provided ``optimizer`` is a :class:`~pytorch_lightning.core.optimizer.LightningOptimizer` object wrapping your own optimizer
configured in your :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`.
You can access your own optimizer with ``optimizer.optimizer``. However, if you use your own optimizer
to perform a step, Lightning won't be able to support accelerators, precision and profiling for you.

.. testcode:: python

    # function hook in LightningModule
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)


    # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
    # To access it, do the following.
    # However, it won't work on TPU, AMP, etc...
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer = optimizer.optimizer
        optimizer.step(closure=optimizer_closure)

-----


Bring your own Custom Learning Rate Schedulers
==============================================

Lightning allows using custom learning rate schedulers that aren't available in `PyTorch natively <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
One good example is `Timm Schedulers <https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py>`_. When using custom learning rate schedulers
relying on a different API from Native PyTorch ones, you should override the :meth:`~pytorch_lightning.core.module.LightningModule.lr_scheduler_step` with your desired logic.
If you are using native PyTorch schedulers, there is no need to override this hook since Lightning will handle it automatically by default.

.. code-block:: python

    from timm.scheduler import TanhLRScheduler


    def configure_optimizers(self):
        optimizer = ...
        scheduler = TanhLRScheduler(optimizer, ...)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


.. _configure_gradient_clipping:

Configure Gradient Clipping
===========================

To configure custom gradient clipping, consider overriding
the :meth:`~pytorch_lightning.core.module.LightningModule.configure_gradient_clipping` method.
Attributes ``gradient_clip_val`` and ``gradient_clip_algorithm`` from Trainer will be passed in the
respective arguments here and Lightning will handle gradient clipping for you. In case you want to set
different values for your arguments of your choice and let Lightning handle the gradient clipping, you can
use the inbuilt :meth:`~pytorch_lightning.core.module.LightningModule.clip_gradients` method and pass
the arguments along with your optimizer.

.. warning::
    Make sure to not override :meth:`~pytorch_lightning.core.module.LightningModule.clip_gradients`
    method. If you want to customize gradient clipping, consider using
    :meth:`~pytorch_lightning.core.module.LightningModule.configure_gradient_clipping` method.

For example, here we will apply gradient clipping only to the gradients associated with optimizer A.

.. testcode:: python

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        if optimizer_idx == 0:
            # Lightning will handle the gradient clipping
            self.clip_gradients(
                optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
            )

Here we configure gradient clipping differently for optimizer B.

.. testcode:: python

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        if optimizer_idx == 0:
            # Lightning will handle the gradient clipping
            self.clip_gradients(
                optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
            )
        elif optimizer_idx == 1:
            self.clip_gradients(
                optimizer, gradient_clip_val=gradient_clip_val * 2, gradient_clip_algorithm=gradient_clip_algorithm
            )


Total Stepping Batches
======================

You can use built-in trainer property :paramref:`~pytorch_lightning.trainer.trainer.Trainer.estimated_stepping_batches` to compute
total number of stepping batches for the complete training. The property is computed considering gradient accumulation factor and
distributed setting into consideration so you don't have to derive it manually. One good example where this can be helpful is while using
:class:`~torch.optim.lr_scheduler.OneCycleLR` scheduler, which requires pre-computed ``total_steps`` during initialization.

.. code-block:: python

    def configure_optimizers(self):
        optimizer = ...
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]
