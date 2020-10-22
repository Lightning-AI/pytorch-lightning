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

* Disable automatic optimization in Trainer:  Trainer(automatic_optimization=False)
* Drop or ignore the optimizer_idx argument
* Use `self.manual_backward(loss)` instead of `loss.backward()` to automatically scale your loss

.. code-block:: python

    def training_step(self, batch, batch_idx, optimizer_idx):
        # ignore optimizer_idx
        (opt_g, opt_d) = self.optimizers()

        # do anything you want
        loss_a = ...

        # use self.backward which will also handle scaling the loss when using amp
        self.manual_backward(loss_a, opt_g)
        opt_g.step()
        opt_g.zero_grad()

        # do anything you want
        loss_b = ...

        # pass in any args that loss.backward() normally takes
        self.manual_backward(loss_b, opt_d, retain_graph=True)
        self.manual_backward(loss_b, opt_d)
        opt_d.step()
        opt_d.zero_grad()

.. note:: This is only recommended for experts who need ultimate flexibility

Manual optimization does not yet support accumulated gradients but will be live in 1.1.0

------

Automatic optimization
======================
With Lightning most users don't have to think about when to call .backward(), .step(), .zero_grad(), since
Lightning automates that for you.

Under the hood Lightning does the following:

.. code-block:: python

    for epoch in epochs:
        for batch id data:
            loss = model.training_step(batch, batch_idx, ...)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for scheduler in scheduler:
            scheduler.step()

In the case of multiple optimizers, Lightning does the following:

.. code-block:: python

    for epoch in epochs:
      for batch in data:
         for opt in optimizers:
            disable_grads_for_other_optimizers()
            train_step(opt)
            opt.step()

      for scheduler in scheduler:
         scheduler.step()


Learning rate scheduling
------------------------
Every optimizer you use can be paired with any `LearningRateScheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.

.. testcode::

   # no LR scheduler
   def configure_optimizers(self):
      return Adam(...)

   # Adam + LR scheduler
   def configure_optimizers(self):
      optimizer = Adam(...)
      scheduler = LambdaLR(optimizer, ...)
      return [optimizer], [scheduler]

   # The ReduceLROnPlateau scheduler requires a monitor
   def configure_optimizers(self):
      return {
          'optimizer': Adam(...),
          'scheduler': ReduceLROnPlateau(optimizer, ...),
          'monitor': 'metric_to_track'
      }

   # Two optimizers each with a scheduler
   def configure_optimizers(self):
      optimizer1 = Adam(...)
      optimizer2 = SGD(...)
      scheduler1 = LambdaLR(optimizer1, ...)
      scheduler2 = LambdaLR(optimizer2, ...)
      return [optimizer1, optimizer2], [scheduler1, scheduler2]

   # Alternatively
   def configure_optimizers(self):
      optimizer1 = Adam(...)
      optimizer2 = SGD(...)
      scheduler1 = ReduceLROnPlateau(optimizer1, ...)
      scheduler2 = LambdaLR(optimizer2, ...)
      return (
          {'optimizer': optimizer1, 'lr_scheduler': scheduler1, 'monitor': 'metric_to_track'},
          {'optimizer': optimizer2, 'lr_scheduler': scheduler2},
      )

   # Same as above with additional params passed to the first scheduler
   def configure_optimizers(self):
      optimizers = [Adam(...), SGD(...)]
      schedulers = [
         {
            'scheduler': ReduceLROnPlateau(optimizers[0], ...),
            'monitor': 'metric_to_track',
            'interval': 'epoch',
            'frequency': 1,
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

      for scheduler in scheduler:
         scheduler.step()

----------

Step optimizers at arbitrary intervals
--------------------------------------
To do more interesting things with your optimizers such as learning rate warm-up or odd scheduling,
override the :meth:`optimizer_step` function.

For example, here step optimizer A every 2 batches and optimizer B every 4 batches

.. testcode::

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step()

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
      optimizer.zero_grad()

    # Alternating schedule for optimizer steps (ie: GANs)
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update generator opt every 2 steps
        if optimizer_i == 0:
            if batch_nb % 2 == 0 :
                optimizer.step()
                optimizer.zero_grad()

        # update discriminator opt every 4 steps
        if optimizer_i == 1:
            if batch_nb % 4 == 0 :
                optimizer.step()
                optimizer.zero_grad()

        # ...
        # add as many optimizers as you want

Here we add a learning-rate warm up

.. testcode::

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step()
        optimizer.zero_grad()

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
