.. list-table:: reg. user 1.5
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``trainer.fit(train_dataloaders=...)``
     - use ``trainer.fit(dataloaders=...)``
     - `PR7431`_

   * - used ``trainer.validate(val_dataloaders...)``
     - use  ``trainer.validate(dataloaders=...)``
     - `PR7431`_

   * - passed ``num_nodes``  to  ``DDPPlugin`` and ``DDPSpawnPlugin``
     - remove them since these parameters are now passed from the ``Trainer``
     - `PR7026`_

   * - passed ``sync_batchnorm`` to ``DDPPlugin`` and ``DDPSpawnPlugin``
     -  remove them since these parameters are now passed from the ``Trainer``
     - `PR7026`_

   * - didn’t provide a ``monitor`` argument to the ``EarlyStopping`` callback and just relied on the default value
     - pass  ``monitor`` as it is now a required argument
     - `PR7907`_

   * - used ``every_n_val_epochs`` in ``ModelCheckpoint``
     - change the argument to ``every_n_epochs``
     - `PR8383`_

   * - used Trainer’s flag ``reload_dataloaders_every_epoch``
     - use pass ``reload_dataloaders_every_n_epochs``
     - `PR5043`_

   * - used Trainer’s flag ``distributed_backend``
     - use ``strategy``
     - `PR8575`_


.. _pr7431: https://github.com/Lightning-AI/lightning/pull/7431
.. _pr7026: https://github.com/Lightning-AI/lightning/pull/7026
.. _pr7907: https://github.com/Lightning-AI/lightning/pull/7907
.. _pr8383: https://github.com/Lightning-AI/lightning/pull/8383
.. _pr5043: https://github.com/Lightning-AI/lightning/pull/5043
.. _pr8575: https://github.com/Lightning-AI/lightning/pull/8575
