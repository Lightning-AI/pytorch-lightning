.. role:: hidden
    :class: hidden-section

Callbacks
=========

Lightning has a callback system to execute arbitrary code. Callbacks should capture NON-ESSENTIAL
logic that is NOT required for your LightningModule to run.

An overall Lightning system should have:

1. Trainer for all engineering
2. LightningModule for all research code.
3. Callbacks for non-essential code.

Example
.. code-block:: python

    import pytorch_lightning as pl

    class MyPrintingCallback(pl.Callback):

        def on_init_start(self, trainer):
            print('Starting to init trainer!')

        def on_init_end(self, trainer):
            print('trainer is init now')

        def on_train_end(self, trainer, pl_module):
            print('do something when training ends')

    # pass to trainer
    trainer = pl.Trainer(callbacks=[MyPrintingCallback()])

We successfully extended functionality without polluting our super clean LightningModule research code

.. automodule:: pytorch_lightning.callbacks
   :exclude-members:
        _del_model,
        _save_model,
        _abc_impl,
        on_epoch_end,
        on_train_end,
        on_epoch_start,
        check_monitor_top_k,
        on_train_start,

