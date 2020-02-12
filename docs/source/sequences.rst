Sequential Data
================
Lightning has built in support for dealing with sequential data.


Packed sequences as inputs
----------------------------
When using PackedSequence, do 2 things:

1. return either a padded tensor in dataset or a list of variable length tensors in the dataloader collate_fn (example above shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

.. code-block:: python

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

.. code-block:: python

    # DEFAULT (single backwards pass per batch)
    trainer = Trainer(truncated_bptt_steps=None)

    # (split batch into sequences of size 2)
    trainer = Trainer(truncated_bptt_steps=2)

.. note:: If you need to modify how the batch is split,
    override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`.

.. note:: Using this feature requires updating your LightningModule's :meth:`pytorch_lightning.core.LightningModule.training_step` to include
    a `hiddens` arg.