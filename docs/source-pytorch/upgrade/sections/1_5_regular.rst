.. list-table:: reg. user 1.5
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``trainer.fit(train_dataloaders=...)`` 
     - use ``trainer.fit(dataloaders=...)``   
     - #7431

   * - used ``trainer.validate(val_dataloaders...)`` 
     - use  ``trainer.validate(dataloaders=...)``  
     - #7431

   * - passed ``num_nodes``  to  ``DDPPlugin`` and ``DDPSpawnPlugin`` 
     - remove them since these parameters are now passed from the ``Trainer``  
     - #7026

   * - passed ``sync_batchnorm`` to ``DDPPlugin`` and ``DDPSpawnPlugin`` 
     -  remove them since these parameters are now passed from the ``Trainer``  
     - #7026

   * - didn’t provide a ``monitor`` argument to the ``EarlyStopping`` callback and just relied on the default value 
     - pass  ``monitor`` as it is now a required argument  
     - #7907

   * - used ``every_n_val_epochs`` in ``ModelCheckpoint`` 
     - change the argument to ``every_n_epochs``  
     - #8383

   * - used Trainer’s flag ``reload_dataloaders_every_epoch`` 
     - use pass ``reload_dataloaders_every_n_epochs``  
     - #5043

   * - used Trainer’s flag ``distributed_backend`` 
     - use ``strategy``  
     - #8575