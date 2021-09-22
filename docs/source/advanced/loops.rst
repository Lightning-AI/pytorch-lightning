.. _loop_customization:

Loop Customization
==================

Loop customization is an experimental feature introduced in Lightning 1.5 that enables users to swap the built-in Lightning loops with custom ones, providing a unprecedented level of flexibility.
Until now, Lightning did not support well some training paradigms like recommendation engine optimization or active learning.
The loop customization feature will not only enable researchers to customize Lightning down to its very core, but also allow one to build new functionalities on top of it.

The training loop in Lightning
------------------------------

Every PyTorch user is familiar with the basic training loop for gradient descent optimization:

.. code-block:: python

    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader):
            x, y = batch
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

At its core, the Lightning Trainer does not do anything different here.
It implements the same loop as shown above except that the research code stays in the LightningModule:

.. code-block:: python

    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader):
            loss = lightning_module.training_step(batch, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

What remains in the Trainer is the loop, zero_grad, backward and optimizer step calls.
These are considered *boilerplate* and get automated by Lightning.

This optimization scheme is very general and applies to the vast majority of deep learning research today.
However, the loops and optimizer calls here remain predetermined in their order and are fully controlled by the Trainer.

Loop customization now enables a new level of control where also the two remaining `for` loops and more can be fully changed or replaced.

Here is how the above training loop can be defined using the new Loop API:

.. code-block:: python

    class FitLoop(Loop):

        def __init__(self):
            self.epoch_loop = EpochLoop()

        def run(self):
            for epoch in range(self.trainer.max_epochs)
                self.advance()

        def advance(self):
            dataloader = lightning_module.train_dataloader()
            self.epoch_loop.run(dataloader)


    class EpochLoop(Loop):

        def run(self, dataloader):
            self.iterator = enumerate(dataloader)
            while True:
                try:
                    self.advance()
                except StopIteration:
                    break

        def advance(self):
            i, batch = next(self.iterator)
            loss = lightning_module.training_step(batch, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


Defining a loop with a class interface instead of hard-coding a raw Python for/while loop has several benefits:

1. you can have full control over the data flow through loops
2. you can add new loops and nest as many of them as they want
3. if needed, the state of a loop can be saved and resumed
4. new hooks can be injected at any point

and much more.
When we have a custom loop defined in a class as shown above, we can attach it to the :code:`Trainer`.

.. code-block:: python

    model = LitModel()
    trainer = Trainer()

    fit_loop = FitLoop()
    # .fit() will use this loop
    trainer.fit_loop = fit_loop
    trainer.fit(model)


Practical example: training_step as a generator
-----------------------------------------------

Lightning supports multiple optimizers and offers a special :code:`training_step` flavor for it, where an extra argument with the current optimizer being used is passed in.
Take as an example the following training step of a DCGAN from the Lightning Bolts repository:

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
            self._training_step_generator = None

        def on_run_start(self, batch, optimizers, batch_idx):
            super().on_run_start(batch, optimizers, batch_idx)
            assert self.trainer.lightning_module.automatic_optimization

            # We request the generator once and save it for later so we can call next() on it.
            self._training_step_generator = self._get_training_step_generator(batch, batch_idx, opt_idx=0)

        def _make_step_fn(self, batch, batch_idx, opt_idx):
            return partial(self._training_step, self._training_step_generator)

        def _get_training_step_generator(self, batch, batch_idx, opt_idx):
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

        def _training_step(self, training_step_generator):
            lightning_module = self.trainer.lightning_module

            # Here, instead of calling lightning_module.training_step()
            # we call next() on the generator!
            training_step_output = next(training_step_generator)

            self.trainer.accelerator.post_training_step()
            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)
            result = ClosureResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)
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
    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=yield_loop)
    trainer.fit(model)  # runs the new loop!

Note that we need to connect it to the :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop` as this is the next higher loop above the optimizer loop.

Finally, we can rewrite the GAN training step using the new yield mechanism:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        real, _ = batch

        # this gets computed in both cases:
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


The Loop base class
-------------------

The :class:`~pytorch_lightning.loops.base.Loop` class is the base for all loops in Lighting just like the LightningModule is the base for all models.
It defines a public interface that each loop implementation must follow, the key ones are:

- :meth:`~pytorch_lightning.loops.base.Loop.advance`: implements the logic of a single iteration in the loop
- :meth:`~pytorch_lightning.loops.base.Loop.done`: a boolean stopping criteria
- :meth:`~pytorch_lightning.loops.base.Loop.reset`: implements a mechanism to reset the loop so it can be restarted

These methods are called by the default implementation of the :meth:`~pytorch_lightning.loops.base.Loop.run` entry point as shown in the (reduced) code excerpt below.

.. code-block:: python

    def run(self, *args, **kwargs):

        self.reset()
        self.on_run_start(*args, **kwargs)

        while not self.done:
            try:
                self.advance(*args, **kwargs)
            except StopIteration:
                break

        output = self.on_run_end()
        return output

Some important observations here: One, the ``run()`` method can define input arguments that get forwarded to some of the other methods that get invoked as part of ``run()``.
Such input arguments typically comprise of one or several iterables over which the loop is supposed to iterate, for example, an iterator over a :class:`~torch.utils.data.DataLoader`.
The reason why the inputs get forwarded is mainly for convenience but implementations are free to change this.
Secondly, ``advance()`` can raise a :class:`StopIteration` to exit the loop early.
This is analogous to a :code:`break` statement in a raw Python ``while`` for example.
Finally, a loop may return an output as part of ``run()``.
As an example, the loop could return a list containing all results produced in each iteration (advance).

Loops can also be nested! That is, a loop may call another one inside of its ``advance()``.


Which other loops does Lightning have and how can they be changed?
------------------------------------------------------------------

The Trainer has four entry points for training, testing and inference, and each method corresponds to a main loop:

+---------------------------------------------------------------+-----------------------------------------------------------------------+-------------------------------------------------------------------------------+
| Entry point                                                   | Trainer attribute                                                     | Loop class                                                                    |
+===============================================================+=======================================================================+===============================================================================+
| :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`        | :attr:`~pytorch_lightning.trainer.trainer.Trainer.fit_loop`           | :class:`~pytorch_lightning.loops.fit_loop.FitLoop`                            |
+---------------------------------------------------------------+-----------------------------------------------------------------------+-------------------------------------------------------------------------------+
| :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`   | :attr:`~pytorch_lightning.trainer.trainer.Trainer.validate_loop`      | :class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop`   |
+---------------------------------------------------------------+-----------------------------------------------------------------------+-------------------------------------------------------------------------------+
| :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`       | :attr:`~pytorch_lightning.trainer.trainer.Trainer.test_loop`          | :class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop`   |
+---------------------------------------------------------------+-----------------------------------------------------------------------+-------------------------------------------------------------------------------+
| :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`    | :attr:`~pytorch_lightning.trainer.trainer.Trainer.predict_loop`       | :class:`~pytorch_lightning.loops.dataloader.prediction_loop.PredictionLoop`   |
+---------------------------------------------------------------+-----------------------------------------------------------------------+-------------------------------------------------------------------------------+

When the user calls :code:`Trainer.<entry-point>`, it redirects to the corresponding :code:`Trainer.loop.run()` which implements the main logic of that particular Lightning loop.
The :meth:`~pytorch_lightning.loops.base.Loop.run` method is part of the base :class:`~pytorch_lightning.loops.base.Loop` class that every loop inherits from (like every model inherits from LightningModule).

Attaching a custom loop to any of these entry points is straightforward.

**Step 1:** Define the loop class by implementing the base interface.

.. code-block:: python

    from pytorch_lightning.loops import Loop


    class CustomLoop(Loop):
        def __init__(self):
            ...

        @property
        def done(self):
            ...

        def advance(self, *args, **kwargs):
            # here goes your custom logic
            ...

Instead of writing an entire new loop, one can also override the behavior of an existing one, for example:

.. code-block:: python

    from pytorch_lightning.loops import FitLoop


    class CustomLoop(FitLoop):
        ...


**Step 2:** Attach the loop to the Trainer and run it.

.. code-block:: python

    loop = CustomLoop()
    trainer = Trainer()

    trainer.fit_loop = loop
    trainer.fit(model)

    # or
    trainer.validate_loop = loop
    trainer.validate(model)

    # or
    trainer.test_loop = loop
    trainer.test(model)

    # or
    trainer.predict_loop = loop
    trainer.predict(model)
