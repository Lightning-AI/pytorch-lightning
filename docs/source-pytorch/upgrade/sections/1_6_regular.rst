.. list-table:: reg. user 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used Trainer’s flag ``terminate_on_nan``
     - set ``detect_anomaly`` instead, which enables detecting anomalies in the autograd engine
     - #9175

   * - used Trainer’s flag ``weights_summary``
     - pass a ``ModelSummary`` callback with ``max_depth`` instead
     - #9699

   * - used Trainer’s flag ``checkpoint_callback``
     - set ``enable_checkpointing``. If you set ``enable_checkpointing=True``, it configures a default ``ModelCheckpoint`` callback if none is provided ``lightning_pytorch.trainer.trainer.Trainer.callbacks.ModelCheckpoint``
     - #9754

   * - used Trainer’s flag ``stochastic_weight_avg``
     - add the ``StochasticWeightAveraging`` callback directly to the list of callbacks, so for example, ``Trainer(..., callbacks=[StochasticWeightAveraging(), ...])``
     - #8989

   * - used Trainer’s flag ``flush_logs_every_n_steps``
     - pass it to the logger init if it is supported for the particular logger
     - #9366

   * - used Trainer’s flag ``max_steps`` to the ``Trainer``, ``max_steps=None`` won't have any effect
     - turn off the limit by passing ``Trainer(max_steps=-1)`` which is the default
     - #9460

   * - used Trainer’s flag ``resume_from_checkpoint="..."``
     - pass the same path to the fit function instead, ``trainer.fit(ckpt_path="...")``
     - #9693

   * - used Trainer’s flag ``log_gpu_memory``, ``gpu_metrics``
     - use the ``DeviceStatsMonitor`` callback instead
     - #9921

   * - used Trainer’s flag ``progress_bar_refresh_rate``
     - set the ``ProgressBar`` callback and set ``refresh_rate`` there, or pass ``enable_progress_bar=False`` to disable the progress bar
     - #9616

   * - called ``LightningModule.summarize()``
     - use the utility function ``pl.utilities.model_summary.summarize(model)``
     - #8513

   * - used the ``LightningModule.model_size`` property
     - use the utility function ``pl.utilities.memory.get_model_size_mb(model)``
     - #8495

   * - relied on the ``on_trai_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     - use ``train_dataloader``
     - #9098

   * - relied on the ``on_val_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     - use ``val_dataloader``
     - #9098

   * - relied on the ``on_test_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     - use ``test_dataloader``
     - #9098

   * - relied on the ``on_predict_dataloader()`` hooks in  ``LightningModule`` and ``LightningDataModule``
     -  use ``predict_dataloader``
     - #9098

   * - implemented the ``on_keyboard_interrupt`` callback hook
     - implement the ``on_exception`` hook, and specify the exception type
     - #9260

   * - relied on the ``TestTubeLogger``
     - Use another logger like ``TensorBoardLogger``
     - #9065

   * - used the basic progress bar ``ProgressBar`` callback
     - use the ``TQDMProgressBar`` callback instead with the same arguments
     - #10134

   * - were using ``GPUStatsMonitor``  callbacks
     - use ``DeviceStatsMonitor`` callback instead
     - #9924

   * - were using ``XLAStatsMonitor`` callbacks
     - use ``DeviceStatsMonitor`` callback instead
     - #9924
