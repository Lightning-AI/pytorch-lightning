.. list-table:: adv. user 1.6
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - passed `prepare_data_per_node` to the `Trainer`
     - set it as a property of `DataHooks`, accessible in the `LightningModule` and `LightningDataModule` instead
     - #8958

   * - used  `process_position` flag
     - specify your  `ProgressBar` callback and set it as `process_position` directly
     - #9222

   * - used distributed training attributes `add_to_queue` and `get_from_queue` in `LightningModule`
     - user the same methods in `DDPStrategy(start_method='spawn')`
     - #9118

   * - called `LightningModule.get_progress_bar_dict`
     - use the utility function `pl.callbacks.progress.base.get_standard_metrics(module.trainer)`
     - #9118

   * - used `LightningModule.on_post_move_to_device`
     - remove it as parameters tying happens automatically without the need of implementing your own logic
     - #9525

   * - relied on  `Trainer.progress_bar_dict`
     - use  `ProgressBarBase.get_metrics`
     - #9118

   * - used `LightningDistributed`
     - rely on the logic in `DDPStrategy(start_method='...')`
     - #9691

   * - used the Accelerator collective API `Accelerator.barrier`, `Accelerator.broadcast`, and `Accelerator.all_gather`
     - call `Strategy` collectives API directly, without going through `Accelerator`
     - #9677

   * - used `pytorch_lightning.core.decorators.parameter_validation`
     - rely on automatic parameters tying with `pytorch_lightning.utilities.params_tying.set_shared_parameters`
     - #9525

   * - used `LearningRateMonitor.lr_sch_names`
     - access them using `LearningRateMonitor.lrs.keys()` which will return the names of all the optimizers, even those without a scheduler.
     - #10066

   * - implemented `DataModule`  `train_transforms`, `val_transforms`, `test_transforms`, `size`, `dims`
     - switch to `LightningDataModule`
     - #8851
