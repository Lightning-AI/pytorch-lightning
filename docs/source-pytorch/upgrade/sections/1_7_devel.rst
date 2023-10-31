.. list-table:: devel 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - Removed the legacy ``Trainer.get_deprecated_arg_names()``
     -
     - `PR14415`_

   * - used the generic method ``Trainer.run_stage``
     - switch to a specific one depending on your purpose ``Trainer.{fit,validate,test,predict}`` .
     - `PR11000`_

   * - used ``rank_zero_only`` from ``pl.utilities.distributed``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``rank_zero_debug`` from ``pl.utilities.distributed``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``rank_zero_info`` from ``pl.utilities.distributed``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``rank_zero_warn`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``rank_zero_deprecation`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``LightningDeprecationWarning`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``LightningDeprecationWarning`` from ``pl.utilities.warnings``
     - import it from ``pl.utilities.rank_zero``
     - `PR11747`_

   * - used ``Trainer.data_parallel_device_ids`` attribute
     - switch it to ``Trainer.device_ids``
     - `PR12072`_

   * - derived it from ``TrainerCallbackHookMixin``
     - use Trainer base class
     - `PR14401`_

   * - used base class ``pytorch_lightning.profiler.BaseProfilerto``
     - switch to use ``pytorch_lightning.profiler.Profiler`` instead
     - `PR12150`_

   * - set distributed backend via the environment variable ``PL_TORCH_DISTRIBUTED_BACKEND``
     - use ``process_group_backend`` in the strategy constructor
     - `PR11745`_

   * - used ``PrecisionPlugin.on_load_checkpoint`` hooks
     - switch to  ``PrecisionPlugin.load_state_dict``
     - `PR11978`_

   * - used ``PrecisionPlugin.on_save_checkpoint`` hooks
     - switch to  ``PrecisionPlugin.load_state_dict``
     - `PR11978`_

   * - used ``Trainer.root_gpu`` attribute
     - use ``Trainer.strategy.root_device.index`` when GPU is used
     - `PR12262`_

   * - used ``Trainer.use_amp`` attribute
     - rely on Torch native AMP
     - `PR12312`_

   * - used ``LightningModule.use_amp`` attribute
     - rely on Torch native AMP
     - `PR12315`_

   * - used Trainer’s attribute ``Trainer.verbose_evaluate``
     - rely on loop constructor  ``EvaluationLoop(verbose=...)``
     - `PR10931`_

   * - used Trainer’s attribute ``Trainer.should_rank_save_checkpoint``
     - it was removed
     - `PR11068`_

   * - derived from ``TrainerOptimizersMixin``
     - rely on ``core/optimizer.py``
     - `PR11155`_

   * - derived from ``TrainerDataLoadingMixin``
     - rely on methods from ``Trainer`` and ``DataConnector``
     - `PR11282`_

   * - used Trainer’s attribute ``Trainer.lightning_optimizers``
     - switch to the ``Strategy`` and its attributes.
     - `PR11444`_

   * - used ``Trainer.call_hook``
     - it was set as a protected method ``Trainer._call_callback_hooks``, ``Trainer._call_lightning_module_hook``, ``Trainer._call_ttp_hook``, ``Trainer._call_accelerator_hook`` and shall not be used.
     - `PR10979`_

   * - used Profiler’s attribute  ``SimpleProfiler.profile_iterable``
     - it was removed
     - `PR12102`_

   * - used Profiler’s attribute  ``AdvancedProfiler.profile_iterable``
     - it was removed
     - `PR12102`_

   * - used the  ``device_stats_monitor.prefix_metric_keys``
     -
     - `PR11254`_

   * - used ``on_train_batch_end(outputs, ...)`` with 2d list with sizes (n_optimizers, tbptt_steps)
     - chang it to (tbptt_steps, n_optimizers). You can update your code by adding the following parameter to your hook signature: ``on_train_batch_end(outputs, ..., new_format=True)``.
     - `PR12182`_

   * - used ``training_epoch_end(outputs)`` with a 3d list with sizes (n_optimizers, n_batches, tbptt_steps)
     - change it to (n_batches, tbptt_steps, n_optimizers). You can update your code by adding the following parameter to your hook signature: ``training_epoch_end(outputs, new_format=True)``.
     - `PR12182`_


.. _pr14415: https://github.com/Lightning-AI/lightning/pull/14415
.. _pr11000: https://github.com/Lightning-AI/lightning/pull/11000
.. _pr11747: https://github.com/Lightning-AI/lightning/pull/11747
.. _pr12072: https://github.com/Lightning-AI/lightning/pull/12072
.. _pr14401: https://github.com/Lightning-AI/lightning/pull/14401
.. _pr12150: https://github.com/Lightning-AI/lightning/pull/12150
.. _pr11745: https://github.com/Lightning-AI/lightning/pull/11745
.. _pr11978: https://github.com/Lightning-AI/lightning/pull/11978
.. _pr12262: https://github.com/Lightning-AI/lightning/pull/12262
.. _pr12312: https://github.com/Lightning-AI/lightning/pull/12312
.. _pr12315: https://github.com/Lightning-AI/lightning/pull/12315
.. _pr10931: https://github.com/Lightning-AI/lightning/pull/10931
.. _pr11068: https://github.com/Lightning-AI/lightning/pull/11068
.. _pr11155: https://github.com/Lightning-AI/lightning/pull/11155
.. _pr11282: https://github.com/Lightning-AI/lightning/pull/11282
.. _pr11444: https://github.com/Lightning-AI/lightning/pull/11444
.. _pr10979: https://github.com/Lightning-AI/lightning/pull/10979
.. _pr12102: https://github.com/Lightning-AI/lightning/pull/12102
.. _pr11254: https://github.com/Lightning-AI/lightning/pull/11254
.. _pr12182: https://github.com/Lightning-AI/lightning/pull/12182
