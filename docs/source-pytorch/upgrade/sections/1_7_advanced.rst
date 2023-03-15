.. list-table:: adv. user 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref
 
   * - used ``DDP2Strategy``
     - switch to ``DDPStrategy``
     - #14026
 
   * - used  ``Trainer.training_type_plugin`` property
     - now use ``Trainer.strategy`` and update the references
     - #11141
 
   * - used any  ``TrainingTypePluginsn``
     - rename them to  ``Strategy``
     - #11120
 
   * - used ``DistributedType``
     - rely on protected ``_StrategyType``
     - #10505
 
   * - used ``DeviceType``
     - rely on protected  ``_AcceleratorType``
     - #10503
 
   * - used ``pl.utiltiies.meta`` functions
     - switch to built-in https://github.com/pytorch/torchdistx support
     - #13868
 
   * - have implemented ``Callback.on_configure_sharded_model`` hook
     - move your implementation to ``Callback.setup``
     - #14834
 
   * - have implemented the ``Callback.on_before_accelerator_backend_setup`` hook
     - move your implementation to ``Callback.setup``
     - #14834
 
   * - have implemented the ``Callback.on_batch_start`` hook
     - move your implementation to ``Callback.on_train_batch_start``
     - #14834
 
   * - have implemented the ``Callback.on_batch_end`` hook
     - move your implementation to ``Callback.on_train_batch_end``
     - #14834
 
   * - have implemented the ``Callback.on_epoch_start`` hook
     - move your implementation  to ``Callback.on_train_epoch_start`` , to ``Callback.on_validation_epoch_start`` , to ``Callback.on_test_epoch_start``
     - #14834
 
   * - have implemented the ``Callback.on_pretrain_routine_{start,end}`` hook
     - move your implementation to ``Callback.on_fit_start``
     - #14834
 
   * - used ``Callback.on_init_start`` hook
     - use ``Callback.on_train_start`` instead
     - #10940
 
   * - used ``Callback.on_init_end``  hook
     - use ``Callback.on_train_start`` instead
     - #10940
 
   * - used Trainer’s attribute ``Trainer.num_processes``
     - it was replaced by  ``Trainer.num_devices``
     - #12388
 
   * - used Trainer’s attribute ``Trainer.gpus``
     - it was replaced by  ``Trainer.num_devices``
     - #12436
 
   * - used Trainer’s attribute ``Trainer.num_gpus``
     - use ``Trainer.num_devices``  instead
     - #12384
 
   * - used Trainer’s attribute ``Trainer.ipus``
     - use  ``Trainer.num_devices``  instead
     - #12386
 
   * - used Trainer’s attribute ``Trainer.tpu_cores``
     - use ``Trainer.num_devices`` instead
     - #12437
 
   * - used  ``Trainer.num_processes`` attribute
     - switch to using ``Trainer.num_devices``
     - #12388
 
   * - used ``LightningIPUModule``
     - it was removed
     - #14830
 
   * - logged with ``LightningLoggerBase.agg_and_log_metrics``
     - switch to ``LightningLoggerBase.log_metrics``
     - #11832
 
   * - used  ``agg_key_funcs``  parameter from ``LightningLoggerBase`` 
     - log metrics explicitly
     - #11871
 
   * - used  ``agg_default_func`` parameters in ``LightningLoggerBase``
     - log metrics explicitly
     - #11871
 
   * - used  ``Trainer.validated_ckpt_path`` attribute
     - rely on generic read-only property ``Trainer.ckpt_path`` which is set when checkpoints are loaded via ``Trainer.validate(````ckpt_path=...)``
     - #11696
 
   * - used  ``Trainer.tested_ckpt_path`` attribute
     - rely on generic read-only property ``Trainer.ckpt_path`` which is set when checkpoints are loaded via ``Trainer.test(````ckpt_path=...)``
     - #11696
 
   * - used  ``Trainer.predicted_ckpt_path`` attribute
     - rely on generic read-only property ``Trainer.ckpt_path``, which is set when checkpoints are loaded via ``Trainer.predict(````ckpt_path=...)``
     - #11696
 
   * - rely on the returned dictionary from  ``Callback.on_save_checkpoint`` 
     - call directly ``Callback.state_dict`` instead
     - #11887