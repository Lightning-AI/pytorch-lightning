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




An interesting property of the abstract loop interface is that it can maintain a state.
It can save its state to a checkpoint through corresponding hooks and if implemented accordingly, resume it's state of exectuion at the appropriate place.
This design is particularly interesting for fault-tolerant training which is an experimental feature released in Lightning v1.5.




FAQ:

**Q:** Why are the loops in Lightning classes and not just simply `for` or `while` loops?
**A:** Partability, state management, complex interactions between loops as object oriented design, advanced users

**Q:** How do I make sure a given LightningModule is compatible with my custom loop?
**A:** To restrict the compatibility of a LightningModule to a particular loop type, we recommend to define a specific class mixin for this purpose.

**Q:** How can I access the Trainer from within a loop?
**A:** There is a :attr:`~pytorch_lightning.loops.base.Loop.trainer` property.
