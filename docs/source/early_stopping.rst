Early stopping
==============

Default behavior
----------------
By default early stopping will be enabled if `'val_loss'`
is found in :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end`'s
return dict. Otherwise training will proceed with early stopping disabled.

Enable Early Stopping
---------------------
There are two ways to enable early stopping.

.. seealso::
    :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. code-block:: python

    # A) Set early_stop_callback to True. Will look for 'val_loss'
    # in validation_epoch_end() return dict. If it is not found an error is raised.
    trainer = Trainer(early_stop_callback=True)

    # B) Or configure your own callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(early_stop_callback=early_stop_callback)

In any case, the callback will fall back to the training metrics (returned in
:meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`,
:meth:`~pytorch_lightning.core.lightning.LightningModule.training_step_end`)
looking for a key to monitor if validation is disabled or
:meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end`
is not defined.


Disable Early Stopping
----------------------
To disable early stopping pass ``False`` to the
:paramref:`~pytorch_lightning.trainer.trainer.Trainer.early_stop_callback`.
Note that ``None`` will not disable early stopping but will lead to the
default behaviour.

.. seealso::
    :class:`~pytorch_lightning.trainer.trainer.Trainer`
