.. list-table:: adv. user 1.7
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used ``DDP2Strategy``
     - switch to ``DDPStrategy``
     - `PR14026`_

   * - used  ``Trainer.training_type_plugin`` property
     - now use ``Trainer.strategy`` and update the references
     - `PR11141`_

   * - used any  ``TrainingTypePluginsn``
     - rename them to  ``Strategy``
     - `PR11120`_

   * - used ``DistributedType``
     - rely on protected ``_StrategyType``
     - `PR10505`_

   * - used ``DeviceType``
     - rely on protected  ``_AcceleratorType``
     - `PR10503`_

   * - used ``pl.utiltiies.meta`` functions
     - switch to built-in https://github.com/pytorch/torchdistx support
     - `PR13868`_

   * - have implemented ``Callback.on_configure_sharded_model`` hook
     - move your implementation to ``Callback.setup``
     - `PR14834`_

   * - have implemented the ``Callback.on_before_accelerator_backend_setup`` hook
     - move your implementation to ``Callback.setup``
     - `PR14834`_

   * - have implemented the ``Callback.on_batch_start`` hook
     - move your implementation to ``Callback.on_train_batch_start``
     - `PR14834`_

   * - have implemented the ``Callback.on_batch_end`` hook
     - move your implementation to ``Callback.on_train_batch_end``
     - `PR14834`_

   * - have implemented the ``Callback.on_epoch_start`` hook
     - move your implementation  to ``Callback.on_train_epoch_start`` , to ``Callback.on_validation_epoch_start`` , to ``Callback.on_test_epoch_start``
     - `PR14834`_

   * - have implemented the ``Callback.on_pretrain_routine_{start,end}`` hook
     - move your implementation to ``Callback.on_fit_start``
     - `PR14834`_

   * - used ``Callback.on_init_start`` hook
     - use ``Callback.on_train_start`` instead
     - `PR10940`_

   * - used ``Callback.on_init_end``  hook
     - use ``Callback.on_train_start`` instead
     - `PR10940`_

   * - used Trainer’s attribute ``Trainer.num_processes``
     - it was replaced by  ``Trainer.num_devices``
     - `PR12388`_

   * - used Trainer’s attribute ``Trainer.gpus``
     - it was replaced by  ``Trainer.num_devices``
     - `PR12436`_

   * - used Trainer’s attribute ``Trainer.num_gpus``
     - use ``Trainer.num_devices``  instead
     - `PR12384`_

   * - used Trainer’s attribute ``Trainer.ipus``
     - use  ``Trainer.num_devices``  instead
     - `PR12386`_

   * - used Trainer’s attribute ``Trainer.tpu_cores``
     - use ``Trainer.num_devices`` instead
     - `PR12437`_

   * - used  ``Trainer.num_processes`` attribute
     - switch to using ``Trainer.num_devices``
     - `PR12388`_

   * - used ``LightningIPUModule``
     - it was removed
     - `PR14830`_

   * - logged with ``LightningLoggerBase.agg_and_log_metrics``
     - switch to ``LightningLoggerBase.log_metrics``
     - `PR11832`_

   * - used  ``agg_key_funcs``  parameter from ``LightningLoggerBase``
     - log metrics explicitly
     - `PR11871`_

   * - used  ``agg_default_func`` parameters in ``LightningLoggerBase``
     - log metrics explicitly
     - `PR11871`_

   * - used  ``Trainer.validated_ckpt_path`` attribute
     - rely on generic read-only property ``Trainer.ckpt_path`` which is set when checkpoints are loaded via ``Trainer.validate(ckpt_path=...)``
     - `PR11696`_

   * - used  ``Trainer.tested_ckpt_path`` attribute
     - rely on generic read-only property ``Trainer.ckpt_path`` which is set when checkpoints are loaded via ``Trainer.test(ckpt_path=...)``
     - `PR11696`_

   * - used  ``Trainer.predicted_ckpt_path`` attribute
     - rely on generic read-only property ``Trainer.ckpt_path``, which is set when checkpoints are loaded via ``Trainer.predict(ckpt_path=...)``
     - `PR11696`_

   * - rely on the returned dictionary from  ``Callback.on_save_checkpoint``
     - call directly ``Callback.state_dict`` instead
     - `PR11887`_


.. _pr14026: https://github.com/Lightning-AI/lightning/pull/14026
.. _pr11141: https://github.com/Lightning-AI/lightning/pull/11141
.. _pr11120: https://github.com/Lightning-AI/lightning/pull/11120
.. _pr10505: https://github.com/Lightning-AI/lightning/pull/10505
.. _pr10503: https://github.com/Lightning-AI/lightning/pull/10503
.. _pr13868: https://github.com/Lightning-AI/lightning/pull/13868
.. _pr14834: https://github.com/Lightning-AI/lightning/pull/14834
.. _pr10940: https://github.com/Lightning-AI/lightning/pull/10940
.. _pr12388: https://github.com/Lightning-AI/lightning/pull/12388
.. _pr12436: https://github.com/Lightning-AI/lightning/pull/12436
.. _pr12384: https://github.com/Lightning-AI/lightning/pull/12384
.. _pr12386: https://github.com/Lightning-AI/lightning/pull/12386
.. _pr12437: https://github.com/Lightning-AI/lightning/pull/12437
.. _pr14830: https://github.com/Lightning-AI/lightning/pull/14830
.. _pr11832: https://github.com/Lightning-AI/lightning/pull/11832
.. _pr11871: https://github.com/Lightning-AI/lightning/pull/11871
.. _pr11696: https://github.com/Lightning-AI/lightning/pull/11696
.. _pr11887: https://github.com/Lightning-AI/lightning/pull/11887
