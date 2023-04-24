*******************
Save Callback state
*******************

Some callbacks require internal state in order to function properly. You can optionally
choose to persist your callback's state as part of model checkpoint files using
:meth:`~lightning.pytorch.callbacks.Callback.state_dict` and :meth:`~lightning.pytorch.callbacks.Callback.load_state_dict`.
Note that the returned state must be able to be pickled.

When your callback is meant to be used only as a singleton callback then implementing the above two hooks is enough
to persist state effectively. However, if passing multiple instances of the callback to the Trainer is supported, then
the callback must define a :attr:`~lightning.pytorch.callbacks.Callback.state_key` property in order for Lightning
to be able to distinguish the different states when loading the callback state. This concept is best illustrated by
the following example.

.. testcode::

    class Counter(Callback):
        def __init__(self, what="epochs", verbose=True):
            self.what = what
            self.verbose = verbose
            self.state = {"epochs": 0, "batches": 0}

        @property
        def state_key(self) -> str:
            # note: we do not include `verbose` here on purpose
            return f"Counter[what={self.what}]"

        def on_train_epoch_end(self, *args, **kwargs):
            if self.what == "epochs":
                self.state["epochs"] += 1

        def on_train_batch_end(self, *args, **kwargs):
            if self.what == "batches":
                self.state["batches"] += 1

        def load_state_dict(self, state_dict):
            self.state.update(state_dict)

        def state_dict(self):
            return self.state.copy()


    # two callbacks of the same type are being used
    trainer = Trainer(callbacks=[Counter(what="epochs"), Counter(what="batches")])

A Lightning checkpoint from this Trainer with the two stateful callbacks will include the following information:

.. code-block::

    {
        "state_dict": ...,
        "callbacks": {
            "Counter{'what': 'batches'}": {"batches": 32, "epochs": 0},
            "Counter{'what': 'epochs'}": {"batches": 0, "epochs": 2},
            ...
        }
    }

The implementation of a :attr:`~lightning.pytorch.callbacks.Callback.state_key` is essential here. If it were missing,
Lightning would not be able to disambiguate the state for these two callbacks, and :attr:`~lightning.pytorch.callbacks.Callback.state_key`
by default only defines the class name as the key, e.g., here ``Counter``.
