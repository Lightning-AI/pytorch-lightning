:orphan:

Loops (Advanced)
================

.. _persisting loop state:

Persisting the State of Loops
-----------------------------

.. note::

    This is an experimental feature and is not activated by default.
    Set the environment variable `PL_FAULT_TOLERANT_TRAINING = 1` to enable saving the progress of loops.
    Read more about :doc:`fault-tolerant training <../advanced/fault_tolerant_training>`.

A powerful property of the class-based loop interface is that it can own an internal state.
Loop instances can save their state to the checkpoint through corresponding hooks and if implemented accordingly, resume the state of execution at the appropriate place.
This design is particularly interesting for fault-tolerant training which is an experimental feature released in Lightning v1.5.

The two hooks :meth:`~pytorch_lightning.loops.base.Loop.on_save_checkpoint` and :meth:`~pytorch_lightning.loops.base.Loop.on_load_checkpoint` function very similarly to how LightningModules and Callbacks save and load state.

.. code-block:: python

    def on_save_checkpoint(self):
        state_dict["iteration"] = self.iteration
        return state_dict


    def on_load_checkpoint(self, state_dict):
        self.iteration = state_dict["iteration"]

When the Trainer is restarting from a checkpoint (e.g., through :code:`trainer.fit(ckpt_path=...)`), the loop exposes a boolean attribute :attr:`~pytorch_lightning.loops.base.Loop.restarting`.
Based around the value of this variable, the user can write the loop in such a way that it can restart from an arbitrary point given the state loaded from the checkpoint.
For example, the implementation of the :meth:`~pytorch_lightning.loops.base.Loop.reset` method could look like this given our previous example:

.. code-block:: python

    def reset(self):
        if not self.restarting:
            self.iteration = 0
