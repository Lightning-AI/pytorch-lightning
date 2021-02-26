.. _optimizers:

************
Optimization
************

Lightning offers two modes for managing the optimization process:

- automatic optimization (AutoOpt)
- manual optimization

For the majority of research cases, **automatic optimization** will do the right thing for you and it is what
most users should use.

For advanced/expert users who want to do esoteric optimization schedules or techniques, use **manual optimization**.

------

Manual optimization
===================
For advanced research topics like reinforcement learning, sparse coding, or GAN research, it may be desirable
to manually manage the optimization process. To do so, do the following:

* Override your LightningModule ``automatic_optimization`` property to return ``False``
* Drop or ignore the optimizer_idx argument
* Use `self.manual_backward(loss)` instead of `loss.backward()`.

.. note:: This is only recommended for experts who need ultimate flexibility. Lightning will handle only precision and accelerators logic. The users are left with zero_grad, accumulated_grad_batches, model toggling, etc..

.. warning:: Before 1.2, ``optimzer.step`` was calling ``zero_grad`` internally. From 1.2, it is left to the users expertize.

.. tip:: To perform ``accumulate_grad_batches`` with one optimizer, you can do as such.

.. tip:: ``self.optimizers()`` will return ``LightningOptimizer`` objects. You can access your own optimizer with ``optimizer.optimizer``. However, if you use your own optimizer to perform a step, Lightning won't be able to support accelerators and precision for you.


.. code-block:: python

    def training_step(batch, batch_idx, optimizer_idx):
        opt = self.optimizers()

        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()

        # accumulate gradient batches
        if batch_idx % 2 == 0:
            opt.zero_grad()


.. tip:: It is a good practice to provide the optimizer with a ``closure`` function that performs a ``forward`` and ``backward`` pass of your model. It is optional for most optimizers, but makes your code compatible if you switch to an optimizer which requires a closure.

Here is the same example as above using a ``closure``.

.. code-block:: python

    def training_step(batch, batch_idx, optimizer_idx):
        opt = self.optimizers()

        def forward_and_backward():
            loss = self.compute_loss(batch)
            self.manual_backward(loss)

        opt.step(closure=forward_and_backward)

        # accumulate gradient batches
        if batch_idx % 2 == 0:
            opt.zero_grad()


.. code-block:: python

    # Scenario for a GAN.

    def training_step(...):
        opt_gen, opt_dis = self.optimizers()

        # compute generator loss
        loss_gen = self.compute_generator_loss(...)

        # zero_grad needs to be called before backward
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        opt_gen.step()

        # compute discriminator loss
        loss_dis = self.compute_discriminator_loss(...)

        # zero_grad needs to be called before backward
        opt_dis.zero_grad()
        self.manual_backward(loss_dis)
        opt_dis.step()


.. note:: ``LightningOptimizer`` provides a ``toggle_model`` function as a ``@context_manager`` for advanced users. It can be useful when performing gradient accumulation with several optimizers or training in a distributed setting.

Here is an explanation of what it does:

Considering the current optimizer as A and all other optimizers as B.
Toggling means that all parameters from B exclusive to A will have their ``requires_grad`` attribute set to ``False``. Their original state will be restored when exiting the context manager.

When performing gradient accumulation, there is no need to perform grad synchronization during the accumulation phase.
Setting ``sync_grad`` to ``False`` will block this synchronization and improve your training speed.

Here is an example on how to use it:

.. code-block:: python


    # Scenario for a GAN with gradient accumulation every 2 batches and optimized for multiple gpus.

    def training_step(self, batch, batch_idx, ...):
        opt_gen, opt_dis = self.optimizers()

        accumulated_grad_batches = batch_idx % 2 == 0

        # compute generator loss
        def closure_gen():
            loss_gen = self.compute_generator_loss(...)
            self.manual_backward(loss_gen)
            if accumulated_grad_batches:
                opt_gen.zero_grad()

        with opt_gen.toggle_model(sync_grad=accumulated_grad_batches):
            opt_gen.step(closure=closure_gen)

        def closure_dis():
            loss_dis = self.compute_discriminator_loss(...)
            self.manual_backward(loss_dis)
            if accumulated_grad_batches:
                opt_dis.zero_grad()

        with opt_dis.toggle_model(sync_grad=accumulated_grad_batches):
            opt_dis.step(closure=closure_dis)

------

Automatic optimization
======================
With Lightning most users don't have to think about when to call .backward(), .step(), .zero_grad(), since
Lightning automates that for you.

Under the hood Lightning does the following:

.. code-block:: python

    for epoch in epochs:
        for batch in data:
            loss = model.training_step(batch, batch_idx, ...)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for scheduler in schedulers:
            scheduler.step()

In the case of multiple optimizers, Lightning does the following:

.. code-block:: python

    for epoch in epochs:
      for batch in data:
         for opt in optimizers:
            disable_grads_for_other_optimizers()
            train_step(opt)
            opt.step()

      for scheduler in schedulers:
         scheduler.step()


Learning rate scheduling
------------------------
Every optimizer you use can be paired with any `LearningRateScheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
In the basic use-case, the scheduler (or multiple schedulers) should be returned as the second output from the ``.configure_optimizers``
method:

.. testcode::

   # no LR scheduler
   def configure_optimizers(self):
      return Adam(...)

   # Adam + LR scheduler
   def configure_optimizers(self):
      optimizer = Adam(...)
      scheduler = LambdaLR(optimizer, ...)
      return [optimizer], [scheduler]

   # Two optimizers each with a scheduler
   def configure_optimizers(self):
      optimizer1 = Adam(...)
      optimizer2 = SGD(...)
      scheduler1 = LambdaLR(optimizer1, ...)
      scheduler2 = LambdaLR(optimizer2, ...)
      return [optimizer1, optimizer2], [scheduler1, scheduler2]

When there are schedulers in which the ``.step()`` method is conditioned on a metric value (for example the
:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler), Lightning requires that the output
from ``configure_optimizers`` should be dicts, one for each optimizer, with the keyword ``monitor``
set to metric that the scheduler should be conditioned on.

.. testcode::

   # The ReduceLROnPlateau scheduler requires a monitor
   def configure_optimizers(self):
      return {
          'optimizer': Adam(...),
          'lr_scheduler': ReduceLROnPlateau(optimizer, ...),
          'monitor': 'metric_to_track'
      }

   # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
   def configure_optimizers(self):
      optimizer1 = Adam(...)
      optimizer2 = SGD(...)
      scheduler1 = ReduceLROnPlateau(optimizer1, ...)
      scheduler2 = LambdaLR(optimizer2, ...)
      return (
          {'optimizer': optimizer1, 'lr_scheduler': scheduler1, 'monitor': 'metric_to_track'},
          {'optimizer': optimizer2, 'lr_scheduler': scheduler2},
      )

.. note::
    Metrics can be made availble to condition on by simply logging it using ``self.log('metric_to_track', metric_val)``
    in your lightning module.

By default, all schedulers will be called after each epoch ends. To change this behaviour, a scheduler configuration should be
returned as a dict which can contain the following keywords:

* ``scheduler`` (required): the actual scheduler object
* ``monitor`` (optional): metric to condition
* ``interval`` (optional): either ``epoch`` (default) for stepping after each epoch ends or ``step`` for stepping
  after each optimization step
* ``frequency`` (optional): how many epochs/steps should pass between calls to ``scheduler.step()``. Default is 1,
  corresponding to updating the learning rate after every epoch/step.
* ``strict`` (optional): if set to ``True`` will enforce that value specified in ``monitor`` is available while trying
  to call ``scheduler.step()``, and stop training if not found. If ``False`` will only give a warning and continue training
  (without calling the scheduler).
* ``name`` (optional): if using the :class:`~pytorch_lightning.callbacks.LearningRateMonitor` callback to monitor the
  learning rate progress, this keyword can be used to specify a specific name the learning rate should be logged as.

.. testcode::

   # Same as the above example with additional params passed to the first scheduler
   # In this case the ReduceLROnPlateau will step after every 10 processed batches
   def configure_optimizers(self):
      optimizers = [Adam(...), SGD(...)]
      schedulers = [
         {
            'scheduler': ReduceLROnPlateau(optimizers[0], ...),
            'monitor': 'metric_to_track',
            'interval': 'step',
            'frequency': 10,
            'strict': True,
         },
         LambdaLR(optimizers[1], ...)
      ]
      return optimizers, schedulers

----------

Use multiple optimizers (like GANs)
-----------------------------------
To use multiple optimizers return > 1 optimizers from :meth:`pytorch_lightning.core.LightningModule.configure_optimizers`

.. testcode::

   # one optimizer
   def configure_optimizers(self):
      return Adam(...)

   # two optimizers, no schedulers
   def configure_optimizers(self):
      return Adam(...), SGD(...)

   # Two optimizers, one scheduler for adam only
   def configure_optimizers(self):
      return [Adam(...), SGD(...)], {'scheduler': ReduceLROnPlateau(), 'monitor': 'metric_to_track'}

Lightning will call each optimizer sequentially:

.. code-block:: python

   for epoch in epochs:
      for batch in data:
         for opt in optimizers:
            train_step(opt)
            opt.step()

      for scheduler in schedulers:
         scheduler.step()

----------

Step optimizers at arbitrary intervals
--------------------------------------
To do more interesting things with your optimizers such as learning rate warm-up or odd scheduling,
override the :meth:`optimizer_step` function.

For example, here step optimizer A every 2 batches and optimizer B every 4 batches

.. testcode::

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
      optimizer.zero_grad()

    # Alternating schedule for optimizer steps (ie: GANs)
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update generator opt every 2 steps
        if optimizer_idx == 0:
            if batch_nb % 2 == 0 :
               optimizer.step(closure=closure)

        # update discriminator opt every 4 steps
        if optimizer_idx == 1:
            if batch_nb % 4 == 0 :
               optimizer.step(closure=closure)

Here we add a learning-rate warm up

.. testcode::

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step(closure=closure)

.. note:: The default ``optimizer_step`` is relying on the internal ``LightningOptimizer`` to properly perform a step. It handles TPUs, AMP, accumulate_grad_batches, zero_grad, and much more ...

.. testcode::

    # function hook in LightningModule
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
      optimizer.step(closure=closure)

.. note:: To access your wrapped Optimizer from ``LightningOptimizer``, do as follow.

.. testcode::

    # function hook in LightningModule
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):

      # `optimizer is a ``LightningOptimizer`` wrapping the optimizer.
      # To access it, do as follow:
      optimizer = optimizer.optimizer

      # run step. However, it won't work on TPU, AMP, etc...
      optimizer.step(closure=closure)


----------

Using the closure functions for optimization
--------------------------------------------

When using optimization schemes such as LBFGS, the `second_order_closure` needs to be enabled. By default, this function is defined by wrapping the `training_step` and the backward steps as follows

.. testcode::

    def second_order_closure(pl_module, split_batch, batch_idx, opt_idx, optimizer, hidden):
        # Model training step on a given batch
        result = pl_module.training_step(split_batch, batch_idx, opt_idx, hidden)

        # Model backward pass
        pl_module.backward(result, optimizer, opt_idx)

        # on_after_backward callback
        pl_module.on_after_backward(result.training_step_output, batch_idx, result.loss)

        return result

    # This default `second_order_closure` function can be enabled by passing it directly into the `optimizer.step`
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update params
        optimizer.step(second_order_closure)
