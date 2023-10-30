:orphan:

Arbitrary iterable support
==========================

Python iterables are objects that can be iterated or looped over. Examples of iterables in Python include lists and dictionaries.
In PyTorch, a :class:`torch.utils.data.DataLoader` is also an iterable which typically retrieves data from a :class:`torch.utils.data.Dataset` or :class:`torch.utils.data.IterableDataset`.

The :class:`~lightning.pytorch.trainer.trainer.Trainer` works with arbitrary iterables, but most people will use a :class:`torch.utils.data.DataLoader` as the iterable to feed data to the model.

.. _multiple-dataloaders:

Multiple Iterables
------------------

In addition to supporting arbitrary iterables, the ``Trainer`` also supports arbitrary collections of iterables. Some examples of this are:

.. code-block:: python

    return DataLoader(...)
    return list(range(1000))

    # pass loaders as a dict. This will create batches like this:
    # {'a': batch_from_loader_a, 'b': batch_from_loader_b}
    return {"a": DataLoader(...), "b": DataLoader(...)}

    # pass loaders as list. This will create batches like this:
    # [batch_from_dl_1, batch_from_dl_2]
    return [DataLoader(...), DataLoader(...)]

    # {'a': [batch_from_dl_1, batch_from_dl_2], 'b': [batch_from_dl_3, batch_from_dl_4]}
    return {"a": [dl1, dl2], "b": [dl3, dl4]}

Lightning automatically collates the batches from multiple iterables based on a "mode". This is done with our
:class:`~lightning.pytorch.utilities.combined_loader.CombinedLoader` class.
The list of modes available can be found by looking at the :paramref:`~lightning.pytorch.utilities.combined_loader.CombinedLoader.mode` documentation.

By default, the ``"max_size_cycle"`` mode is used during training and the ``"sequential"`` mode is used during validation, testing, and prediction.
To choose a different mode, you can use the :class:`~lightning.pytorch.utilities.combined_loader.CombinedLoader` class directly with your mode of choice:

.. code-block:: python

    from lightning.pytorch.utilities import CombinedLoader

    iterables = {"a": DataLoader(), "b": DataLoader()}
    combined_loader = CombinedLoader(iterables, mode="min_size")
    model = ...
    trainer = Trainer()
    trainer.fit(model, combined_loader)


Currently, the ``trainer.predict`` method only supports the ``"sequential"`` mode, while ``trainer.fit`` method does not support it.
Support for this feature is tracked in this `issue <https://github.com/Lightning-AI/lightning/issues/16830>`__.

Note that when using the ``"sequential"`` mode, you need to add an additional argument ``dataloader_idx`` to some specific hooks.
Lightning will `raise an error <https://github.com/Lightning-AI/lightning/pull/16837>`__ informing you of this requirement.

Using LightningDataModule
-------------------------

You can set more than one :class:`~torch.utils.data.DataLoader` in your :class:`~lightning.pytorch.core.datamodule.LightningDataModule` using its DataLoader hooks
and Lightning will use the correct one.

.. testcode::

    class DataModule(LightningDataModule):
        def train_dataloader(self):
            # any iterable or collection of iterables
            return DataLoader(self.train_dataset)

        def val_dataloader(self):
            # any iterable or collection of iterables
            return [DataLoader(self.val_dataset_1), DataLoader(self.val_dataset_2)]

        def test_dataloader(self):
            # any iterable or collection of iterables
            return DataLoader(self.test_dataset)

        def predict_dataloader(self):
            # any iterable or collection of iterables
            return DataLoader(self.predict_dataset)

Using LightningModule Hooks
---------------------------

The exact same code as above works when overriding :class:`~lightning.pytorch.core.LightningModule`

Passing the iterables to the Trainer
------------------------------------

The same support for arbitrary iterables, or collection of iterables applies to the dataloader arguments of
:meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`, :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.test`, :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`
