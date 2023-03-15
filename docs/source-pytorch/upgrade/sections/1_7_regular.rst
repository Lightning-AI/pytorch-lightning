.. list-table:: reg. user 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - have wrapped your loggers with ``LoggerCollection``
     - directly pass a list of loggers to the Trainer and access the list via the ``trainer.loggers`` attribute.
     - #12147

   * - used ``Trainer.lr_schedulers``
     - access ``trainer.lr_scheduler_configs`` instead, which contains dataclasses instead of dictionaries.
     - #11443

   * - used ``neptune-client`` API in the ``NeptuneLogger``
     - upgrade to the latest API
     - #14727

   * - used  ``LightningDataModule.on_save`` hook
     - use  ``LightningDataModule.on_save_checkpoint`` instead
     - #11887

   * - used  ``LightningDataModule.on_load_checkpoint`` hook
     - use  ``LightningDataModule.on_load_checkpoint`` hook instead
     - #11887

   * - used  ``LightningModule.on_hpc_load`` hook
     - switch to general purpose hook ``LightningModule.on_load_checkpoint``
     - #14315

   * - used  ``LightningModule.on_hpc_save`` hook
     - switch to general purpose hook ``LightningModule.on_save_checkpoint``
     - #14315

   * - used Trainer’s flag ``weights_save_path``
     - use directly ``dirpath`` argument in the ``ModelCheckpoint`` callback.
     - #14424

   * - used Trainer’s property ``Trainer.weights_save_path`` is dropped
     - ...
     - #14424
