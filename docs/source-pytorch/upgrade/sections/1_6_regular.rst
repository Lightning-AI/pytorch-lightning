.. list-table:: reg. user 1.6
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used Trainer’s flag ``terminate_on_nan``
     - set ``detect_anomaly`` instead, which enables detecting anomalies in the autograd engine
     - `PR9175`_

   * - used Trainer’s flag ``weights_summary``
     - pass a ``ModelSummary`` callback with ``max_depth`` instead
     - `PR9699`_

   * - used Trainer’s flag ``checkpoint_callback``
     - set ``enable_checkpointing``. If you set ``enable_checkpointing=True``, it configures a default ``ModelCheckpoint`` callback if none is provided ``lightning.pytorch.trainer.trainer.Trainer.callbacks.ModelCheckpoint``
     - `PR9754`_

   * - used Trainer’s flag ``stochastic_weight_avg``
     - add the ``StochasticWeightAveraging`` callback directly to the list of callbacks, so for example, ``Trainer(..., callbacks=[StochasticWeightAveraging(), ...])``
     - `PR8989`_

   * - used Trainer’s flag ``flush_logs_every_n_steps``
     - pass it to the logger init if it is supported for the particular logger
     - `PR9366`_

   * - used Trainer’s flag ``max_steps`` to the ``Trainer``, ``max_steps=None`` won't have any effect
     - turn off the limit by passing ``Trainer(max_steps=-1)`` which is the default
     - `PR9460`_

   * - used Trainer’s flag ``resume_from_checkpoint="..."``
     - pass the same path to the fit function instead, ``trainer.fit(ckpt_path="...")``
     - `PR9693`_

   * - used Trainer’s flag ``log_gpu_memory``, ``gpu_metrics``
     - use the ``DeviceStatsMonitor`` callback instead
     - `PR9921`_

   * - used Trainer’s flag ``progress_bar_refresh_rate``
     - set the ``ProgressBar`` callback and set ``refresh_rate`` there, or pass ``enable_progress_bar=False`` to disable the progress bar
     - `PR9616`_

   * - called ``LightningModule.summarize()``
     - use the utility function ``pl.utilities.model_summary.summarize(model)``
     - `PR8513`_

   * - used the ``LightningModule.model_size`` property
     - use the utility function ``pl.utilities.memory.get_model_size_mb(model)``
     - `PR8495`_

   * - relied on the ``on_train_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     - use ``train_dataloader``
     - `PR9098`_

   * - relied on the ``on_val_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     - use ``val_dataloader``
     - `PR9098`_

   * - relied on the ``on_test_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     - use ``test_dataloader``
     - `PR9098`_

   * - relied on the ``on_predict_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     -  use ``predict_dataloader``
     - `PR9098`_

   * - implemented the ``on_keyboard_interrupt`` callback hook
     - implement the ``on_exception`` hook, and specify the exception type
     - `PR9260`_

   * - relied on the ``TestTubeLogger``
     - Use another logger like ``TensorBoardLogger``
     - `PR9065`_

   * - used the basic progress bar ``ProgressBar`` callback
     - use the ``TQDMProgressBar`` callback instead with the same arguments
     - `PR10134`_

   * - were using ``GPUStatsMonitor``  callbacks
     - use ``DeviceStatsMonitor`` callback instead
     - `PR9924`_

   * - were using ``XLAStatsMonitor`` callbacks
     - use ``DeviceStatsMonitor`` callback instead
     - `PR9924`_


.. _pr9175: https://github.com/Lightning-AI/lightning/pull/9175
.. _pr9699: https://github.com/Lightning-AI/lightning/pull/9699
.. _pr9754: https://github.com/Lightning-AI/lightning/pull/9754
.. _pr8989: https://github.com/Lightning-AI/lightning/pull/8989
.. _pr9366: https://github.com/Lightning-AI/lightning/pull/9366
.. _pr9460: https://github.com/Lightning-AI/lightning/pull/9460
.. _pr9693: https://github.com/Lightning-AI/lightning/pull/9693
.. _pr9921: https://github.com/Lightning-AI/lightning/pull/9921
.. _pr9616: https://github.com/Lightning-AI/lightning/pull/9616
.. _pr8513: https://github.com/Lightning-AI/lightning/pull/8513
.. _pr8495: https://github.com/Lightning-AI/lightning/pull/8495
.. _pr9098: https://github.com/Lightning-AI/lightning/pull/9098
.. _pr9260: https://github.com/Lightning-AI/lightning/pull/9260
.. _pr9065: https://github.com/Lightning-AI/lightning/pull/9065
.. _pr10134: https://github.com/Lightning-AI/lightning/pull/10134
.. _pr9924: https://github.com/Lightning-AI/lightning/pull/9924
