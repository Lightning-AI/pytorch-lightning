.. list-table:: adv. user 1.6
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - passed ``prepare_data_per_node`` to the ``Trainer``
     - set it as a property of ``DataHooks``, accessible in the ``LightningModule`` and ``LightningDataModule`` instead
     - `PR8958`_

   * - used  ``process_position`` flag
     - specify your  ``ProgressBar`` callback and set it as ``process_position`` directly
     - `PR9222`_

   * - used distributed training attributes ``add_to_queue`` and ``get_from_queue`` in ``LightningModule``
     - user the same methods in ``DDPStrategy(start_method='spawn')``
     - `PR9118`_

   * - called ``LightningModule.get_progress_bar_dict``
     - use the utility function ``pl.callbacks.progress.base.get_standard_metrics(module.trainer)``
     - `PR9118`_

   * - used ``LightningModule.on_post_move_to_device``
     - remove it as parameters tying happens automatically without the need of implementing your own logic
     - `PR9525`_

   * - relied on  ``Trainer.progress_bar_dict``
     - use  ``ProgressBarBase.get_metrics``
     - `PR9118`_

   * - used ``LightningDistributed``
     - rely on the logic in ``DDPStrategy(start_method='...')``
     - `PR9691`_

   * - used the Accelerator collective API ``Accelerator.barrier``, ``Accelerator.broadcast``, and ``Accelerator.all_gather``
     - call ``Strategy`` collectives API directly, without going through ``Accelerator``
     - `PR9677`_

   * - used ``pytorch_lightning.core.decorators.parameter_validation``
     - rely on automatic parameters tying with ``pytorch_lightning.utilities.params_tying.set_shared_parameters``
     - `PR9525`_

   * - used ``LearningRateMonitor.lr_sch_names``
     - access them using ``LearningRateMonitor.lrs.keys()`` which will return the names of all the optimizers, even those without a scheduler.
     - `PR10066`_

   * - implemented ``DataModule``  ``train_transforms``, ``val_transforms``, ``test_transforms``, ``size``, ``dims``
     - switch to ``LightningDataModule``
     - `PR8851`_


.. _pr8958: https://github.com/Lightning-AI/lightning/pull/8958
.. _pr9222: https://github.com/Lightning-AI/lightning/pull/9222
.. _pr9118: https://github.com/Lightning-AI/lightning/pull/9118
.. _pr9525: https://github.com/Lightning-AI/lightning/pull/9525
.. _pr9691: https://github.com/Lightning-AI/lightning/pull/9691
.. _pr9677: https://github.com/Lightning-AI/lightning/pull/9677
.. _pr10066: https://github.com/Lightning-AI/lightning/pull/10066
.. _pr8851: https://github.com/Lightning-AI/lightning/pull/8851
