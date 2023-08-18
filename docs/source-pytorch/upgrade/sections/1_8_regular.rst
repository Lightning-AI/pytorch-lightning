.. list-table:: reg. user 1.8
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``seed_everything_default=None`` in ``LightningCLI``
     - set ``seed_everything_default=False`` instead
     - `PR12804`_

   * - used ``Trainer.reset_train_val_dataloaders()``
     - call ``Trainer.fit_loop.setup_data()``
     - `PR12184`_

   * - imported ``pl.core.lightning``
     - import ``pl.core.module`` instead
     - `PR12740`_


.. _pr12804: https://github.com/Lightning-AI/lightning/pull/12804
.. _pr12184: https://github.com/Lightning-AI/lightning/pull/12184
.. _pr12740: https://github.com/Lightning-AI/lightning/pull/12740
