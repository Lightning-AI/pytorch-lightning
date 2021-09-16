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

When a user calls :code:`Trainer.method`, it redirects to the corresponding :code:`Trainer.loop.run()` which implements the main logic of that particular Lightning loop.
Think of it as the start of a Python :code:`while` loop.
The :meth:`~pytorch_lightning.loops.base.Loop.run` method is part of the base :class:`~pytorch_lightning.loops.base.Loop` class that every loop inherits from (like every model inherits from LightningModule).


The Loop base class
-------------------

The :class:`~pytorch_lightning.loops.base.Loop` class is the base for all loops in Lighting just like the LightningModule is the base for all models.
It defines a public interface that each loop implementation must follow, the key ones are:

- :class:`~pytorch_lightning.loops.base.Loop.advance`: implements the logic of a single iteration in the loop
- :class:`~pytorch_lightning.loops.base.Loop.done`: a boolean stopping criteria
- :class:`~pytorch_lightning.loops.base.Loop.reset`: implements a mechanism to reset the loop so it can be restarted

These methods are called by the default implementation of the :class:`~pytorch_lightning.loops.base.Loop.run` entry point as shown in the code excerpt below.

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


An interesting property of this abstract loop interface is that it can maintain a state.
It can save its state to a checkpoint through corresponding hooks and if implemented accordingly, resume it's state of exectuion at the appropriate place.
This design is particularly interesting for fault-tolerant training which is an experimental feature released in Lightning v1.5.


FAQ:

**Q:** Why are the loops in Lightning classes and not just simply `for` or `while` loops?
**A:** Partability, state management, complex interactions between loops as object oriented design, advanced users

**Q:** How do I make sure a given LightningModule is compatible with my custom loop?
**A:** To restrict the compatibility of a LightningModule to a particular loop type, we recommend to define a specific class mixin for this purpose.

**Q:** How can I access the Trainer from within a loop?
**A:** There is a :attr:`~pytorch_lightning.loops.base.Loop.trainer` property.
