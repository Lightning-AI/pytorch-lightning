:orphan:

Accessing DataLoaders
=====================

In the case that you require access to the :class:`torch.utils.data.DataLoader` or :class:`torch.utils.data.Dataset` objects, DataLoaders for each step can be accessed
via the trainer properties :meth:`~lightning.pytorch.trainer.trainer.Trainer.train_dataloader`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.val_dataloaders`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.test_dataloaders`, and
:meth:`~lightning.pytorch.trainer.trainer.Trainer.predict_dataloaders`.

.. code-block:: python

    dataloaders = trainer.train_dataloader
    dataloaders = trainer.val_dataloaders
    dataloaders = trainer.test_dataloaders
    dataloaders = trainer.predict_dataloaders

These properties will match exactly what was returned in your ``*_dataloader`` hooks or passed to the ``Trainer``,
meaning that if you returned a dictionary of dataloaders, these will return a dictionary of dataloaders.

Replacing DataLoaders
---------------------

If you are using a :class:`~lightning.pytorch.utilities.CombinedLoader`. A flattened list of DataLoaders can be accessed by doing:

.. code-block:: python

    from lightning.pytorch.utilities import CombinedLoader

    iterables = {"dl1": dl1, "dl2": dl2}
    combined_loader = CombinedLoader(iterables)
    # access the original iterables
    assert combined_loader.iterables is iterables
    # the `.flattened` property can be convenient
    assert combined_loader.flattened == [dl1, dl2]
    # for example, to do a simple loop
    updated = []
    for dl in combined_loader.flattened:
        new_dl = apply_some_transformation_to(dl)
        updated.append(new_dl)
    # it also allows you to easily replace the dataloaders
    combined_loader.flattened = updated


Reloading DataLoaders During Training
-------------------------------------

Lightning provides two mechanisms for reloading dataloaders during training:

**Automatic reload with** ``reload_dataloaders_every_n_epochs``

Set ``reload_dataloaders_every_n_epochs`` in the Trainer to automatically reload dataloaders at regular intervals:

.. code-block:: python

    trainer = Trainer(reload_dataloaders_every_n_epochs=5)

This is useful when your dataset changes periodically, such as in online learning scenarios.

**Manual reload with** ``trainer.reload_dataloaders()``

For dynamic scenarios like curriculum learning or adaptive training strategies, use
:meth:`~lightning.pytorch.trainer.trainer.Trainer.reload_dataloaders` to trigger a reload
based on training metrics or other conditions:

.. code-block:: python

    class CurriculumCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.callback_metrics.get("train_loss", 1.0) < 0.5:
                # Update datamodule parameters
                trainer.datamodule.difficulty_level += 1
                # Trigger reload for next epoch
                trainer.reload_dataloaders(train=True, val=True)

Or directly from your LightningModule:

.. code-block:: python

    class MyModel(LightningModule):
        def on_train_batch_end(self, outputs, batch, batch_idx):
            if self.trainer.callback_metrics.get("train_loss", 1.0) < 0.5:
                self.trainer.datamodule.sequence_length += 10
                self.trainer.reload_dataloaders()

The reload happens at the start of the next epoch, ensuring training state consistency.

