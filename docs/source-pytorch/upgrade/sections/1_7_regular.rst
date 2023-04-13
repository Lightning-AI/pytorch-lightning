.. list-table:: reg. user 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - have wrapped your loggers with ``LoggerCollection``
     - directly pass a list of loggers to the Trainer and access the list via the ``trainer.loggers`` attribute.
     - `PR12147`_

   * - used ``Trainer.lr_schedulers``
     - access ``trainer.lr_scheduler_configs`` instead, which contains dataclasses instead of dictionaries.
     - `PR11443`_

   * - used ``neptune-client`` API in the ``NeptuneLogger``
     - upgrade to the latest API
     - `PR14727`_

   * - used  ``LightningDataModule.on_save`` hook
     - use  ``LightningDataModule.on_save_checkpoint`` instead
     - `PR11887`_

   * - used  ``LightningDataModule.on_load_checkpoint`` hook
     - use  ``LightningDataModule.on_load_checkpoint`` hook instead
     - `PR11887`_

   * - used  ``LightningModule.on_hpc_load`` hook
     - switch to general purpose hook ``LightningModule.on_load_checkpoint``
     - `PR14315`_

   * - used  ``LightningModule.on_hpc_save`` hook
     - switch to general purpose hook ``LightningModule.on_save_checkpoint``
     - `PR14315`_

   * - used Trainer’s flag ``weights_save_path``
     - use directly ``dirpath`` argument in the ``ModelCheckpoint`` callback.
     - `PR14424`_

   * - used Trainer’s property ``Trainer.weights_save_path`` is dropped
     -
     - `PR14424`_


.. _pr12147: https://github.com/Lightning-AI/lightning/pull/12147
.. _pr11443: https://github.com/Lightning-AI/lightning/pull/11443
.. _pr14727: https://github.com/Lightning-AI/lightning/pull/14727
.. _pr11887: https://github.com/Lightning-AI/lightning/pull/11887
.. _pr14315: https://github.com/Lightning-AI/lightning/pull/14315
.. _pr14424: https://github.com/Lightning-AI/lightning/pull/14424
