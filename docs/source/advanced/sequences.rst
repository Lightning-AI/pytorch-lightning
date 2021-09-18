
Sequential Data
================

Truncated Backpropagation Through Time
--------------------------------------
There are times when multiple backwards passes are needed for each batch.
For example, it may save memory to use Truncated Backpropagation Through Time when training RNNs.

Lightning can handle TBTT automatically via this flag.

.. testcode::

    from pytorch_lightning import LightningModule


    class MyModel(LightningModule):
        def __init__(self):
            super().__init__()
            # Important: This property activates truncated backpropagation through time
            # Setting this value to 2 splits the batch into sequences of size 2
            self.truncated_bptt_steps = 2

        # Truncated back-propagation through time
        def training_step(self, batch, batch_idx, hiddens):
            # the training step must be updated to accept a ``hiddens`` argument
            # hiddens are the hiddens from the previous truncated backprop step
            out, hiddens = self.lstm(data, hiddens)
            return {"loss": ..., "hiddens": hiddens}

.. note:: If you need to modify how the batch is split,
    override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`.
