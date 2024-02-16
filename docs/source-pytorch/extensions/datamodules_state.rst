Save DataModule state
=====================
When a checkpoint is created, it asks every DataModule for their state. If your DataModule defines the *state_dict* and *load_state_dict* methods, the checkpoint will automatically track and restore your DataModules.

.. code:: python

    import lightning as L


    class LitDataModule(L.LightningDataModule):
        def state_dict(self):
            # track whatever you want here
            state = {"current_train_batch_index": self.current_train_batch_index}
            return state

        def load_state_dict(self, state_dict):
            # restore the state based on what you tracked in (def state_dict)
            self.current_train_batch_index = state_dict["current_train_batch_index"]
