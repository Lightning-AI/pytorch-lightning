.. list-table:: devel 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - Removed the legacy ``Trainer.get_deprecated_arg_names()``
     - ...
     - #14415

   * - used the generic method ``Trainer.run_stage``
     - switch to a specific one depending on your purpose ``Trainer.{fit,validate,test,predict}`` .
     - #11000

   * - used ``rank_zero_only`` from ``pl.utilities.distributed``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``rank_zero_debug`` from ``pl.utilities.distributed``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``rank_zero_info`` from ``pl.utilities.distributed``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``rank_zero_warn`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``rank_zero_deprecation`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``LightningDeprecationWarning`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``LightningDeprecationWarning`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - #11747

   * - used ``Trainer.data_parallel_device_ids`` attribute
     - switch it to ``Trainer.device_ids``
     - #12072

   * - derived it from ``TrainerCallbackHookMixin``
     - use Trainer base class
     - #14401

   * - used base class ``pytorch_lightning.profiler.BaseProfilerto``
     - switch to use ``pytorch_lightning.profiler.Profiler`` instead
     - #12150

   * - set distributed backend via the environment variable ``PL_TORCH_DISTRIBUTED_BACKEND``
     - use ``process_group_backend`` in the strategy constructor
     - #11745

   * - used ``PrecisionPlugin.on_load_checkpoint`` hooks
     - switch to  ``PrecisionPlugin.load_state_dict``
     - #11978

   * - used ``PrecisionPlugin.on_save_checkpoint`` hooks
     - switch to  ``PrecisionPlugin.load_state_dict``
     - #11978

   * - used ``Trainer.root_gpu`` attribute
     - use ``Trainer.strategy.root_device.index`` when GPU is used
     - #12262

   * - used ``Trainer.use_amp`` attribute
     - rely on Torch native AMP
     - #12312

   * - used ``LightingModule.use_amp`` attribute
     - rely on Torch native AMP
     - #12315

   * - used Trainer’s attribute ``Trainer.verbose_evaluate``
     - rely on loop constructor  ``EvaluationLoop(verbose=...)``
     - #10931

   * - used Trainer’s attribute ``Trainer.should_rank_save_checkpoint``
     - it was removed
     - #11068

   * - derived from ``TrainerOptimizersMixin``
     - rely on ``core/optimizer.py``
     - #11155

   * - derived from ``TrainerDataLoadingMixin``
     - rely on methods from ``Trainer`` and ``DataConnector``
     - #11282

   * - used Trainer’s attribute ``Trainer.lightning_optimizers``
     - switch to the ``Strategy`` and its attributes.
     - #11444

   * - used ``Trainer.call_hook``
     - it was set as a protected method ``Trainer._call_callback_hooks``, ``Trainer._call_lightning_module_hook``, ``Trainer._call_ttp_hook``, ``Trainer._call_accelerator_hook`` and shall not be used.
     - #10979

   * - used Profiler’s attribute  ``SimpleProfiler.profile_iterable``
     - it was removed
     - #12102

   * - used Profiler’s attribute  ``AdvancedProfiler.profile_iterable``
     - it was removed
     - #12102

   * - used the  ``device_stats_monitor.prefix_metric_keys``
     - ...
     - #11254

   * - used ``on_train_batch_end(outputs, ...)`` with 2d list with sizes (n_optimizers, tbptt_steps)
     - chang it to (tbptt_steps, n_optimizers). You can update your code by adding the following parameter to your hook signature: ``on_train_batch_end(outputs, ..., new_format=True)``.
     - #12182

   * - used ``training_epoch_end(outputs)`` with a 3d list with sizes (n_optimizers, n_batches, tbptt_steps)
     - change it to (n_batches, tbptt_steps, n_optimizers). You can update your code by adding the following parameter to your hook signature: ``training_epoch_end(outputs, new_format=True)``.
     - #12182
