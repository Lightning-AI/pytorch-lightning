.. list-table:: reg. user 1.4
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - relied on the ``outputs`` in your  ``LightningModule.on_train_epoch_end`` or ``Callback.on_train_epoch_end`` hooks
     - rely on either ``on_train_epoch_end`` or set outputs as attributes in your ``LightningModule`` instances and access them from the hook
     - #7339

   * - accessed ``Trainer.truncated_bptt_steps``
     - swicth to manual optimization
     - #7323

   * - called  ``LightningModule.write_predictions``  and  ``LightningModule.write_predictions_dict``
     - rely on ``predict_step`` and ``Trainer.predict`` + callbacks to write out predictions
     - #7066

   * - passed the ``period`` argument to the ``ModelCheckpoint`` callback
     - pass the ``every_n_epochs`` argument to the ``ModelCheckpoint`` callback
     - #6146

   * - passed the ``output_filename`` argument to ``Profiler``
     - now pass ``dirpath`` and ``filename``, that is  ``Profiler(dirpath=...., filename=...)``
     - #6621

   * - passed the ``profiled_functions`` argument in  ``PytorchProfiler``
     - now pass the  ``record_functions`` argument
     - #6349

   * - relied on the ``@auto_move_data`` decorator to use the ``LightningModule`` outside of the ``Trainer`` for inference
     - use ``Trainer.predict``
     - #6993

   * - implemented ``on_load_checkpoint`` with a ``checkpoint`` only argument, as in ``Callback.on_load_checkpoint(checkpoint)``
     - now update the signature to include ``pl_module`` and ``trainer``, as in ``Callback.on_load_checkpoint(trainer, pl_module, checkpoint)``
     - #7253

   * - relied on ``pl.metrics``
     - now import separate package ``torchmetrics``
     - https://torchmetrics.readthedocs.io/en/stable

   * - accessed ``datamodule`` attribute of ``LightningModule``, that is ``model.datamodule``
     - now access ``Trainer.datamodule``, that is ``model.trainer.datamodule``
     - #7168
