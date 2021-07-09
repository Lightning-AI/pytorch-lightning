.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

.. _early_stopping:

**************
Early stopping
**************

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_earlystop.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+19-+early+stopping_1.mp4"></video>

|

Stopping an epoch early
=======================
You can stop an epoch early by overriding :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_train_batch_start` to return ``-1`` when some condition is met.

If you do this repeatedly, for every epoch you had originally requested, then this will stop your entire run.

----------

Early stopping based on metric using the EarlyStopping Callback
===============================================================
The
:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
callback can be used to monitor a validation metric and stop the training when no improvement is observed.

To enable it:

- Import :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback.
- Log the metric you want to monitor using :func:`~pytorch_lightning.core.lightning.LightningModule.log` method.
- Init the callback, and set `monitor` to the logged metric of your choice.
- Pass the :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback to the :class:`~pytorch_lightning.trainer.trainer.Trainer` callbacks flag.

.. code-block:: python

    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    def validation_step(...):
        self.log('val_loss', loss)

    trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss')])

You can customize the callbacks behaviour by changing its parameters.

.. testcode::

    early_stop_callback = EarlyStopping(
       monitor='val_accuracy',
       min_delta=0.00,
       patience=3,
       verbose=False,
       mode='max'
    )
    trainer = Trainer(callbacks=[early_stop_callback])


Additional parameters that stop training at extreme points:

- ``stopping_threshold``: Stops training immediately once the monitored quantity reaches this threshold.
  It is useful when we know that going beyond a certain optimal value does not further benefit us.
- ``divergence_threshold``: Stops training as soon as the monitored quantity becomes worse than this threshold.
  When reaching a value this bad, we believe the model cannot recover anymore and it is better to stop early and run with different initial conditions.
- ``check_finite``: When turned on, we stop training if the monitored metric becomes NaN or infinite.

In case you need early stopping in a different part of training, subclass :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
and change where it is called:

.. testcode::

    class MyEarlyStopping(EarlyStopping):

        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer, pl_module)

.. note::
   The :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback runs
   at the end of every validation epoch,
   which, under the default configuration, happen after every training epoch.
   However, the frequency of validation can be modified by setting various parameters
   in the :class:`~pytorch_lightning.trainer.trainer.Trainer`,
   for example :paramref:`~pytorch_lightning.trainer.trainer.Trainer.check_val_every_n_epoch`
   and :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval`.
   It must be noted that the `patience` parameter counts the number of
   validation epochs with no improvement, and not the number of training epochs.
   Therefore, with parameters `check_val_every_n_epoch=10` and `patience=3`, the trainer
   will perform at least 40 training epochs before being stopped.

.. seealso::
    - :class:`~pytorch_lightning.trainer.trainer.Trainer`
    - :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
