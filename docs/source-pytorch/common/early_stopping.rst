.. testsetup:: *

    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

.. _early_stopping:


##############
Early Stopping
##############

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+19-+early+stopping_1.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_earlystop.png
    :width: 400
    :muted:


***********************
Stopping an Epoch Early
***********************

You can stop and skip the rest of the current epoch early by overriding :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_train_batch_start` to return ``-1`` when some condition is met.

If you do this repeatedly, for every epoch you had originally requested, then this will stop your entire training.


**********************
EarlyStopping Callback
**********************

The :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` callback can be used to monitor a metric and stop the training when no improvement is observed.

To enable it:

- Import :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` callback.
- Log the metric you want to monitor using :meth:`~lightning.pytorch.core.LightningModule.log` method.
- Init the callback, and set ``monitor`` to the logged metric of your choice.
- Set the ``mode`` based on the metric needs to be monitored.
- Pass the :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` callback to the :class:`~lightning.pytorch.trainer.trainer.Trainer` callbacks flag.

.. code-block:: python

    from lightning.pytorch.callbacks.early_stopping import EarlyStopping


    class LitModel(LightningModule):
        def validation_step(self, batch, batch_idx):
            loss = ...
            self.log("val_loss", loss)


    model = LitModel()
    trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(model)

You can customize the callbacks behaviour by changing its parameters.

.. testcode::

    early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    trainer = Trainer(callbacks=[early_stop_callback])


Additional parameters that stop training at extreme points:

- ``stopping_threshold``: Stops training immediately once the monitored quantity reaches this threshold.
  It is useful when we know that going beyond a certain optimal value does not further benefit us.
- ``divergence_threshold``: Stops training as soon as the monitored quantity becomes worse than this threshold.
  When reaching a value this bad, we believes the model cannot recover anymore and it is better to stop early and run with different initial conditions.
- ``check_finite``: When turned on, it stops training if the monitored metric becomes NaN or infinite.
- ``check_on_train_epoch_end``: When turned on, it checks the metric at the end of a training epoch. Use this only when you are monitoring any metric logged within
  training-specific hooks on epoch-level.


In case you need early stopping in a different part of training, subclass :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`
and change where it is called:

.. testcode::

    class MyEarlyStopping(EarlyStopping):
        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer)

.. note::
   The :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` callback runs
   at the end of every validation epoch by default. However, the frequency of validation
   can be modified by setting various parameters in the :class:`~lightning.pytorch.trainer.trainer.Trainer`,
   for example :paramref:`~lightning.pytorch.trainer.trainer.Trainer.check_val_every_n_epoch`
   and :paramref:`~lightning.pytorch.trainer.trainer.Trainer.val_check_interval`.
   It must be noted that the ``patience`` parameter counts the number of
   validation checks with no improvement, and not the number of training epochs.
   Therefore, with parameters ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer
   will perform at least 40 training epochs before being stopped.
