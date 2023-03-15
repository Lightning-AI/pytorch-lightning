.. list-table:: adv. user 1.5
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``self.log(sync_dist_op=...)``
     - use ``self.log(reduce_fx=...)`` instead. Passing ``"mean"`` will still work, but it also takes a callable
     - #7891

   * - used the argument ``model`` from ``pytorch_lightning.utilities.model_helper.is_overridden``
     - use ``instance`` instead
     - #7918

   * - returned values from ``training_step`` that had ``.grad`` defined (e.g., a loss) and expected ``.detach()`` to be called for you
     - call ``.detach()`` manually
     - #7994

   * - imported ``pl.utilities.distributed.rank_zero_warn``
     - import ``pl.utilities.rank_zero.rank_zero_warn``
     -

   * - relied on ``DataModule.has_prepared_data`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_setup_fit`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_setup_validate`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_setup_test`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_setup_predict`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_teardown_fit`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_teardown_validate`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_teardown_test`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - relied on ``DataModule.has_teardown_predict`` attribute
     - manage data lifecycle in customer methods
     - #7657

   * - used ``DDPPlugin.task_idx``
     - use ``DDPStrategy.local_rank``
     - #8203

   * - used ``Trainer.disable_validation``
     - use the condition ``not Trainer.enable_validation``
     - #8291
