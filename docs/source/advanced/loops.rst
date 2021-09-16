.. _loop_customization:

Loop Customization
==================

Loop customization is an experimental feature introduced in Lightning 1.5 that enables advanced users to write new training logic or modify existing behavior in Lightning's training-, evaluation-, or prediction loops.
By advanced users we mean users that are familiar with the major components under the Trainer and how they interact with the LightningModule.

In this advanced user guide we will learn about

- how the Trainer runs a loop,
- the Loop base class,
- the default loop implementations and subloops Lightning has,
- how Lightning defines a tree structure of loops and subloops,
- how you can create a custom loop for a new training step flavor,
- and how you can connect the custom loop and run it.

Most importantly, we will also provide guidelines when to use loop customization and when NOT to use it.


Trainer entry points for loops
------------------------------

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

When the user calls :code:`Trainer.method`, it redirects to the corresponding :code:`Trainer.loop.run()` which implements the main logic of that particular Lightning loop.
Think of it as the start of a Python :code:`while` loop.
The :meth:`~pytorch_lightning.loops.base.Loop.run` method is part of the base :class:`~pytorch_lightning.loops.base.Loop` class that every loop inherits from (like every model inherits from LightningModule).


The Loop base class
-------------------

The :class:`~pytorch_lightning.loops.base.Loop` class is the base for all loops in Lighting just like the LightningModule is the base for all models.
It defines a public interface that each loop implementation must follow, the key ones are:

- :meth:`~pytorch_lightning.loops.base.Loop.advance`: implements the logic of a single iteration in the loop
- :meth:`~pytorch_lightning.loops.base.Loop.done`: a boolean stopping criteria
- :meth:`~pytorch_lightning.loops.base.Loop.reset`: implements a mechanism to reset the loop so it can be restarted

These methods are called by the default implementation of the :meth:`~pytorch_lightning.loops.base.Loop.run` entry point as shown in the code excerpt below.

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

Some important observations here: One, the `run()` method can define input arguments that get forwarded to some of the other methods that get invoked as part of `run()`.
Such input arguments typically comprise of one or several iterables over which the loop is suppose to iterate, for example, an iterator over a :class:`~torch.utils.data.DataLoader`.
The reason why the inputs get forwarded is mainly for convenience but implementations are free to change this.
Secondly, `advance()` can raise a :class:`StopIteration` to exit the loop early.
This is analogeous to a :code:`break` statement in a raw Python `while`-loop for example.
Finally, a loop may return an output as part of `run()`.
This output could for example be a list containing all results produced in each iteration (advance) of the loop.

Loops can also be nested! That is, a loop may call another one inside of its `advance()`.

Default loop implementations
----------------------------

The training loop in Lightning is called *fit loop* and is actually a combination of several loops.
Here is what the structure would look like in plain Python:

.. code-block:: python

    # FitLoop
    for epoch in range(max_epochs):

        # TrainingEpochLoop
        for batch_idx, batch in enumerate(train_dataloader):

            # TrainingBatchLoop
            for  split_batch in tbptt_split(batch):

                # OptimizerLoop
                for optimizer_idx, opt in enumerate(optimizers):

                    loss = lightning_module.training_step(batch, batch_idx, optimizer_idx)
                    ...

            # ValidationEpochLoop
            for batch_idx, batch in enumerate(val_dataloader):
                lightning_module.validation_step(batch, batch_idx, optimizer_idx)
                ...


Each of these :code:`for`-loops represents a class implementing the :class:`~pytorch_lightning.loops.base.Loop` interface.

FitLoop
^^^^^^^

The :class:`~pytorch_lightning.loops.fit_loop.FitLoop` is the top-level loop where training starts.
It simply counts the epochs and iterates from one to the next by calling :code:`TrainingEpochLoop.run()` in its :code:`advance()` method.

TrainingEpochLoop
^^^^^^^^^^^^^^^^^

The :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop` is the one that iterates over the dataloader that the user returns in their :meth:`~pytorch_lightning.core.lightning.LightningModule.train_dataloader` method.
Its main responsibilities are calling the :code:`*_epoch_start` and :code:`*_epoch_end` hooks, accumulating outputs if the user request them in one of these hooks, and running validation at the requested interval.
The validation is carried out by yet another loop, :class:`~pytorch_lightning.loops.epoch.validation_epoch_loop.ValidationEpochLoop`.

In the :code:`run()` method, the training epoch loop could in theory simply call the :code:`LightningModule.training_step` already and perform the optimization.
However, Lightning has built-in support for automatic optimization with multiple optimizers and on top of that also supports truncated back-propagation through time (TODO: add link).
For this reason there are actually two more loops nested under :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop`.

TrainingBatchLoop
^^^^^^^^^^^^^^^^^

The responsibility of the :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop` is to split a batch given by the :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop` along the time-dimension and iterate over the list of splits.
It also keeps track of the hidden state *hiddens* returned by the training step.
By default, when truncated back-propagation through time (TBPTT) is turned off, this loop does not do anything except redirect the call to the :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop`.
You can read more about TBPTT here (TODO: add link).

OptimizerLoop
^^^^^^^^^^^^^

The :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` iterates over one or multiple optimizers and for each one it calls the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` method with the batch, the current batch index and the optimizer index if multiple optimizers are requested.
It is the leaf node in the tree of loops and performs the actual optimization (forward, zero grad, backward, optimizer step).


Custom loops
------------

The training loop in Lightning is very general and does not make many assumptions about how deep learning models are trained.
In almost all use cases the user can do all their research inside the LightningModule alone and will never have to write a custom loop.
This is very much the intended way and the whole reason why Lightning exists in the first place; to separate the loop boilerplate code from the actual research that we care about the most.

However, for more exotic research it may not always be as easy to implement a new algorithm with just the hooks available in Lightning.
Maybe there is a need for a hook in a particular place but it does not exist in Lightning? Or some hooks need to be executed in a different order?
Or maybe the way a loop pre-fetches data needs to be changed to optimize performance?

Loop customization provides an interface that enables such modifications deep inside the Lightning Trainer.
This level of customization is meant for expert Lightning users who are already familiar with the many interactions between internal components.

.. warning:: Loop customization is an experimental feature and is subject to change.

A custom loop, like every loop, needs to implement the base :class:`~pytorch_lightning.loops.base.Loop` interface.

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

    class CustomFitLoop(FitLoop):
        ...


A custom loop can be attached in two ways.
If the loop is aiming to replace one of the top-level loop for fit, validate, test or predict, then set it directly on the trainer like so:

.. code-block:: python

    loop = CustomLoop()
    trainer = Trainer()

    trainer.fit_loop = loop
    # or
    trainer.validate_loop = loop
    # or
    trainer.test_loop = loop
    # or
    trainer.predict_loop = loop

The second way is when a custom loop is meant to replace one of the subloops of a top-level loop.
In this case one can use the :meth:`~pytorch_lightning.loops.base.Loop.connect` method of a parent loop to connect/replace a child loop.
For example:

.. code-block:: python

    loop = CustomLoop()
    trainer = Trainer()

    trainer.fit_loop.connect(epoch_loop=loop)

To illustrate the power of loop customization we will look at a relatively simple custom loop that converts the training_step hook to a generator.

Example: YieldLoop
^^^^^^^^^^^^^^^^^^

Here we will build a simple example of a custom loop that enables us to write a new flavor of a training step, where the training step actually becomes a generator and instead of returning losses for optimization, we yield them!
**Note:** This assumes knowledge of generators in Pythoin and the :code:`yield` mechanism.

Imagine we had a LightningModule training step definition like this:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        # do something with optimizer 0
        loss0 = ....

        yield loss0

        # do something with optimizer 1 that requires loss0
        loss1 = self.foo(loss0)
        yield loss1


Normally, we would implement a training step with a signature :code:`training_step(self, batch, batch_idx, optimizer_idx)` and then conditionally compute either :code:`loss0` or :code:`loss1` depending on the current optimizer index and return that loss at the end.
But if the computation of say :code:`loss1` depends on :code:`loss0` or another quantity computed for the first optimizer, we would have to recompute the value for :code:`optimizer_idx = 1` and that is wasteful.

With the training step as a generator as shown above however, we are able to retain the local variables across training_step boundaries when we switch from one optimizer to the next.
The alternative to this would be *manual optimization* where the same can be achieved, but with the generator loop we can still get all benefits of manual optimization without having to call backward or zero grad ourselves.

In order to enable returning a generator from a training step, we need a custom loop!
This will be a subclass of the existing :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` and then be attached to the :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop`.

.. code-block:: python


    from functools import partial
    from pytorch_lightning.loops import Loop, OptimizerLoop
    from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
    from pytorch_lightning.loops.utilities import _build_training_step_kwargs


    class YieldLoop(OptimizerLoop):
        def __init__(self):
            super().__init__()
            self._training_step_generator = None

        def connect(self, **kwargs):
            raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

        def on_run_start(self, batch, optimizers, batch_idx):
            super().on_run_start(batch, optimizers, batch_idx)
            assert self.trainer.lightning_module.automatic_optimization

            # We request the generator once and save it for later so we can call next() on it.
            self._training_step_generator = self._get_training_step_generator(batch, batch_idx, opt_idx=0)

        def _make_step_fn(self, batch, batch_idx, opt_idx):
            return partial(self._training_step, self._training_step_generator)

        def _get_training_step_generator(self, batch, batch_idx, opt_idx):
            step_kwargs = _build_training_step_kwargs(
                self.trainer.lightning_module, self.trainer.optimizers, batch, batch_idx, opt_idx, hiddens=None
            )

            # Here we are basically calling lightning_module.training_step() and this returns a generator!
            generator = self.trainer.accelerator.training_step(step_kwargs)
            return generator

        def _training_step(self, training_step_generator):
            lightning_module = self.trainer.lightning_module

            with self.trainer.profiler.profile("model_forward"):
                lightning_module._current_fx_name = "training_step"
                with self.trainer.profiler.profile("training_step"):

                    # Here, instead of calling lightning_module.training_step() we call next() on the generator!
                    training_step_output = next(training_step_generator)
                    self.trainer.accelerator.post_training_step()

                training_step_output = self.trainer.call_hook("training_step_end", training_step_output)
                result = ClosureResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)
            return result


As we can see, not much work needs to be done to enable our generator training step.
The new loop is called :code:`YieldLoop` and contains a reference to the generator returned by the :code:`training_step`.
On every new run (over the optimizers) we call the :code:`training_step` method on the LightningModule which is supposed to return a generator because it contains :code:`yield` statements.
There must be as many :code:`yield` statements as there are optimizers.

Given this new loop, here is how you connect it to the Trainer:

.. code-block:: python
    model = LitModel()
    trainer = Trainer()

    yield_loop = YieldLoop()
    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=yield_loop)

    trainer.fit(model)  # runs the new loop!

Note that we need to connect it to the :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop` and we are replacing the default optimizer loop that Lightning provides.
