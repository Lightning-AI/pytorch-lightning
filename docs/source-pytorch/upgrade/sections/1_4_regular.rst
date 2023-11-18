.. list-table:: reg. user 1.4
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - relied on the ``outputs`` in your  ``LightningModule.on_train_epoch_end`` or ``Callback.on_train_epoch_end`` hooks
     - rely on either ``on_train_epoch_end`` or set outputs as attributes in your ``LightningModule`` instances and access them from the hook
     - `PR7339`_

   * - accessed ``Trainer.truncated_bptt_steps``
     - switch to manual optimization
     - `PR7323`_

   * - called  ``LightningModule.write_predictions``  and  ``LightningModule.write_predictions_dict``
     - rely on ``predict_step`` and ``Trainer.predict`` + callbacks to write out predictions
     - `PR7066`_

   * - passed the ``period`` argument to the ``ModelCheckpoint`` callback
     - pass the ``every_n_epochs`` argument to the ``ModelCheckpoint`` callback
     - `PR6146`_

   * - passed the ``output_filename`` argument to ``Profiler``
     - now pass ``dirpath`` and ``filename``, that is  ``Profiler(dirpath=...., filename=...)``
     - `PR6621`_

   * - passed the ``profiled_functions`` argument in  ``PytorchProfiler``
     - now pass the  ``record_functions`` argument
     - `PR6349`_

   * - relied on the ``@auto_move_data`` decorator to use the ``LightningModule`` outside of the ``Trainer`` for inference
     - use ``Trainer.predict``
     - `PR6993`_

   * - implemented ``on_load_checkpoint`` with a ``checkpoint`` only argument, as in ``Callback.on_load_checkpoint(checkpoint)``
     - now update the signature to include ``pl_module`` and ``trainer``, as in ``Callback.on_load_checkpoint(trainer, pl_module, checkpoint)``
     - `PR7253`_

   * - relied on ``pl.metrics``
     - now import separate package ``torchmetrics``
     - `torchmetrics`_

   * - accessed ``datamodule`` attribute of ``LightningModule``, that is ``model.datamodule``
     - now access ``Trainer.datamodule``, that is ``model.trainer.datamodule``
     - `PR7168`_


.. _torchmetrics: https://torchmetrics.readthedocs.io/en/stable
.. _pr7339: https://github.com/Lightning-AI/lightning/pull/7339
.. _pr7323: https://github.com/Lightning-AI/lightning/pull/7323
.. _pr7066: https://github.com/Lightning-AI/lightning/pull/7066
.. _pr6146: https://github.com/Lightning-AI/lightning/pull/6146
.. _pr6621: https://github.com/Lightning-AI/lightning/pull/6621
.. _pr6349: https://github.com/Lightning-AI/lightning/pull/6349
.. _pr6993: https://github.com/Lightning-AI/lightning/pull/6993
.. _pr7253: https://github.com/Lightning-AI/lightning/pull/7253
.. _pr7168: https://github.com/Lightning-AI/lightning/pull/7168
