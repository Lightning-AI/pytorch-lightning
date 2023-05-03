.. list-table:: adv. user 1.5
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``self.log(sync_dist_op=...)``
     - use ``self.log(reduce_fx=...)`` instead. Passing ``"mean"`` will still work, but it also takes a callable
     - `PR7891`_

   * - used the argument ``model`` from ``pytorch_lightning.utilities.model_helper.is_overridden``
     - use ``instance`` instead
     - `PR7918`_

   * - returned values from ``training_step`` that had ``.grad`` defined (e.g., a loss) and expected ``.detach()`` to be called for you
     - call ``.detach()`` manually
     - `PR7994`_

   * - imported ``pl.utilities.distributed.rank_zero_warn``
     - import ``pl.utilities.rank_zero.rank_zero_warn``
     -

   * - relied on ``DataModule.has_prepared_data`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_setup_fit`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_setup_validate`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_setup_test`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_setup_predict`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_teardown_fit`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_teardown_validate`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_teardown_test`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - relied on ``DataModule.has_teardown_predict`` attribute
     - manage data lifecycle in customer methods
     - `PR7657`_

   * - used ``DDPPlugin.task_idx``
     - use ``DDPStrategy.local_rank``
     - `PR8203`_

   * - used ``Trainer.disable_validation``
     - use the condition ``not Trainer.enable_validation``
     - `PR8291`_


.. _pr7891: https://github.com/Lightning-AI/lightning/pull/7891
.. _pr7918: https://github.com/Lightning-AI/lightning/pull/7918
.. _pr7994: https://github.com/Lightning-AI/lightning/pull/7994
.. _pr7657: https://github.com/Lightning-AI/lightning/pull/7657
.. _pr8203: https://github.com/Lightning-AI/lightning/pull/8203
.. _pr8291: https://github.com/Lightning-AI/lightning/pull/8291
