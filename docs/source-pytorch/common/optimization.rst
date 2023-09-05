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

For more advanced use cases like multiple optimizers, esoteric optimization schedules or techniques, use **manual optimization**.

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
                loss = model.training_step(batch, batch_idx)
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(closure)

        lr_scheduler.step()

As can be seen in the code snippet above, Lightning defines a closure with ``training_step()``, ``optimizer.zero_grad()``
and ``loss.backward()`` for the optimization. This mechanism is in place to support optimizers which operate on the
output of the closure (e.g. the loss) or need to call the closure several times (e.g. :class:`~torch.optim.LBFGS`).

Should you still require the flexibility of calling ``.zero_grad()``, ``.backward()``, or ``.step()`` yourself, you can
always switch to :ref:`manual optimization <manual_optimization>`.
Manual optimization is required if you wish to work with multiple optimizers.


.. _gradient_accumulation:

Gradient Accumulation
=====================

.. include:: ../common/gradient_accumulation.rst


Access your Own Optimizer
=========================

The provided ``optimizer`` is a :class:`~lightning.pytorch.core.optimizer.LightningOptimizer` object wrapping your own optimizer
configured in your :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`.
You can access your own optimizer with ``optimizer.optimizer``. However, if you use your own optimizer
to perform a step, Lightning won't be able to support accelerators, precision and profiling for you.

.. testcode:: python

    # function hook in LightningModule
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
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
        optimizer_closure,
    ):
        optimizer = optimizer.optimizer
        optimizer.step(closure=optimizer_closure)

-----


Bring your own Custom Learning Rate Schedulers
==============================================

Lightning allows using custom learning rate schedulers that aren't available in `PyTorch natively <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
One good example is `Timm Schedulers <https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py>`_. When using custom learning rate schedulers
relying on a different API from Native PyTorch ones, you should override the :meth:`~lightning.pytorch.core.LightningModule.lr_scheduler_step` with your desired logic.
If you are using native PyTorch schedulers, there is no need to override this hook since Lightning will handle it automatically by default.

.. code-block:: python

    from timm.scheduler import TanhLRScheduler


    def configure_optimizers(self):
        optimizer = ...
        scheduler = TanhLRScheduler(optimizer, ...)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


.. _configure_gradient_clipping:

Configure Gradient Clipping
===========================

To configure custom gradient clipping, consider overriding
the :meth:`~lightning.pytorch.core.LightningModule.configure_gradient_clipping` method.
The attributes ``gradient_clip_val`` and ``gradient_clip_algorithm`` from Trainer will be passed in the
respective arguments here and Lightning will handle gradient clipping for you. In case you want to set
different values for your arguments of your choice and let Lightning handle the gradient clipping, you can
use the inbuilt :meth:`~lightning.pytorch.core.LightningModule.clip_gradients` method and pass
the arguments along with your optimizer.

.. warning::
    Make sure to not override :meth:`~lightning.pytorch.core.LightningModule.clip_gradients`
    method. If you want to customize gradient clipping, consider using
    :meth:`~lightning.pytorch.core.LightningModule.configure_gradient_clipping` method.

For example, here we will apply a stronger gradient clipping after a certain number of epochs:

.. testcode:: python

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        if self.current_epoch > 5:
            gradient_clip_val = gradient_clip_val * 2

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)


Total Stepping Batches
======================

You can use built-in trainer property :paramref:`~lightning.pytorch.trainer.trainer.Trainer.estimated_stepping_batches` to compute
total number of stepping batches for the complete training. The property is computed considering gradient accumulation factor and
distributed setting into consideration so you don't have to derive it manually. One good example where this can be helpful is while using
:class:`~torch.optim.lr_scheduler.OneCycleLR` scheduler, which requires pre-computed ``total_steps`` during initialization.

.. code-block:: python

    def configure_optimizers(self):
        optimizer = ...
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return optimizer, scheduler
