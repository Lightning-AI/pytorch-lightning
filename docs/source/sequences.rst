.. testsetup:: *

    from torch.utils.data import IterableDataset
    from pytorch_lightning.trainer.trainer import Trainer

Sequential Data
================
Lightning has built in support for dealing with sequential data.


Packed sequences as inputs
----------------------------
When using PackedSequence, do 2 things:

1. return either a padded tensor in dataset or a list of variable length tensors in the dataloader collate_fn (example above shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

.. testcode::

    # For use in dataloader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y

    # In module
    def training_step(self, batch, batch_nb):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)

Truncated Backpropagation Through Time
---------------------------------------
There are times when multiple backwards passes are needed for each batch.
For example, it may save memory to use Truncated Backpropagation Through Time when training RNNs.

Lightning can handle TBTT automatically via this flag.

.. testcode::

    # DEFAULT (single backwards pass per batch)
    trainer = Trainer(truncated_bptt_steps=None)

    # (split batch into sequences of size 2)
    trainer = Trainer(truncated_bptt_steps=2)

.. note:: If you need to modify how the batch is split,
    override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`.

.. note:: Using this feature requires updating your LightningModule's :meth:`pytorch_lightning.core.LightningModule.training_step` to include
    a `hiddens` arg.

Iterable Datasets
---------------------------------------
Lightning supports using IterableDatasets as well as map-style Datasets. IterableDatasets provide a more natural
option when using sequential data.

.. note:: When using an IterableDataset you must set the val_check_interval to 1.0 (the default) or to an int
    (specifying the number of training batches to run before validation) when initializing the Trainer.
    This is due to the fact that the IterableDataset does not have a __len__ and Lightning requires this to calculate
    the validation interval when val_check_interval is less than one.

.. testcode::

    # IterableDataset
    class CustomDataset(IterableDataset):

        def __init__(self, data):
            self.data_source

        def __iter__(self):
            return iter(self.data_source)

    # Setup DataLoader
    def train_dataloader(self):
        seq_data = ['A', 'long', 'time', 'ago', 'in', 'a', 'galaxy', 'far', 'far', 'away']
        iterable_dataset = CustomDataset(seq_data)

        dataloader = DataLoader(dataset=iterable_dataset, batch_size=5)
        return dataloader

.. testcode::

    # Set val_check_interval
    trainer = Trainer(val_check_interval=100)
