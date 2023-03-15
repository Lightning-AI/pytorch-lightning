.. list-table:: reg. user 1.8
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``seed_everything_default=None`` in ``LightningCLI``
     - set ``seed_everything_default=False`` instead
     - #12804

   * - used ``Trainer.reset_train_val_dataloaders()``
     - call ``Trainer.reset_train_dataloaders()`` and ``Trainer.reset_val_dataloaders()`` separately
     - #12184

   * - imported ``pl.core.lightning``
     - import ``pl.core.module`` instead
     - #12740