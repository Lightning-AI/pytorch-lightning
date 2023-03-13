:orphan:

Iterable Datasets
=================

Lightning supports using :class:`~torch.utils.data.IterableDataset` as well as map-style Datasets. IterableDatasets provide a more natural
option when using sequential data.

.. note:: When using an :class:`~torch.utils.data.IterableDataset` you must set the ``val_check_interval`` to 1.0 (the default) or an int
    (specifying the number of training batches to run before each validation loop) when initializing the Trainer. This is
    because the IterableDataset does not have a ``__len__`` and Lightning requires this to calculate the validation
    interval when ``val_check_interval`` is less than one. Similarly, you can set ``limit_{mode}_batches`` to a float or
    an int. If it is set to 0.0 or 0, it will set ``num_{mode}_batches`` to 0, if it is an int, it will set ``num_{mode}_batches``
    to ``limit_{mode}_batches``, if it is set to 1.0 it will run for the whole dataset, otherwise it will throw an exception.
    Here ``mode`` can be train/val/test/predict.

When iterable datasets are used, Lightning will pre-fetch 1 batch (in addition to the current batch) so it can detect
when the training will stop and run validation if necessary.

.. testcode::

    # IterableDataset
    class CustomDataset(IterableDataset):
        def __init__(self, data):
            self.data_source = data

        def __iter__(self):
            return iter(self.data_source)


    # Setup DataLoader
    def train_dataloader(self):
        seq_data = ["A", "long", "time", "ago", "in", "a", "galaxy", "far", "far", "away"]
        iterable_dataset = CustomDataset(seq_data)

        dataloader = DataLoader(dataset=iterable_dataset, batch_size=5)
        return dataloader


.. testcode::

    # Set val_check_interval as an int
    trainer = Trainer(val_check_interval=100)

    # Disable validation: Set limit_val_batches to 0.0 or 0
    trainer = Trainer(limit_val_batches=0.0)

    # Set limit_val_batches as an int
    trainer = Trainer(limit_val_batches=100)
