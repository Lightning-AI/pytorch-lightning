.. list-table:: adv. user 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used the ``pl.lite`` module
     - switch to ``lightning_fabric``
     - #15953

   * - used Trainer’s flag ``strategy='dp'``
     - use DDP with ``strategy='ddp'`` or DeepSpeed instead
     - #16748

   * - implemented ``LightningModule.training_epoch_end`` hooks
     - port your logic to  ``LightningModule.on_training_epoch_end`` hook
     - #16520

   * - implemented ``LightningModule.validation_epoch_end`` hook
     - port your logic to  ``LightningModule.on_validation_epoch_end`` hook
     - #16520

   * - implemented ``LightningModule.test_epoch_end`` hooks
     - port your logic to  ``LightningModule.on_test_epoch_end`` hook
     - #16520

   * - used Trainer’s flag ``multiple_trainloader_mode``
     - switch to  ``CombinedLoader(..., mode=...)`` and set mode directly now
     - #16800

   * - used Trainer’s flag ``move_metrics_to_cpu``
     - implement particular offload logic in your custom metric or turn it on in ``torchmetrics``
     - #16358

   * - used Trainer’s flag ``track_grad_norm``
     - overwrite ``on_before_optimizer_step`` hook and pass the argument directly and ``LightningModule.log_grad_norm()`` hook
     - #16745 #16745

   * - used Trainer’s flag ``replace_sampler_ddp``
     - use  ``use_distributed_sampler``; the sampler gets created not only for the DDP strategies
     -

   * - relied on the ``on_tpu`` argument in ``LightningModule.optimizer_step`` hook
     - switch to manual optimization
     - #16537 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - relied on the ``using_lbfgs`` argument in ``LightningModule.optimizer_step`` hook
     - switch to manual optimization
     - #16538 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - were using ``nvidia/apex`` in any form
     - switch to PyTorch native mixed precision ``torch.amp`` instead
     - #16039 :doc:`Precision <../../common/precision>`

   * - used Trainer’s flag ``using_native_amp``
     - use PyTorch native mixed precision
     - #16039 :doc:`Precision <../../common/precision>`

   * - used Trainer’s flag ``amp_backend``
     - use PyTorch native mixed precision
     - #16039 :doc:`Precision <../../common/precision>`

   * - used Trainer’s flag ``amp_level``
     - use PyTorch native mixed precision
     - #16039 :doc:`Precision <../../common/precision>`

   * - used Trainer’s attribute ``using_native_amp``
     - use PyTorch native mixed precision
     - #16039 :doc:`Precision <../../common/precision>`

   * - used Trainer’s attribute ``amp_backend``
     - use PyTorch native mixed precision
     - #16039 :doc:`Precision <../../common/precision>`

   * - used Trainer’s attribute ``amp_level``
     - use PyTorch native mixed precision
     - #16039 :doc:`Precision <../../common/precision>`

   * - use the ``FairScale`` integration
     - consider using PyTorch's native FSDP implementation or outsourced implementation into own project
     - https://github.com/Lightning-Sandbox/lightning-Fairscale

   * - used ``pl.overrides.fairscale.LightningShardedDataParallel``
     - use native FSDP instead
     - #16400 :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.plugins.precision.fully_sharded_native_amp.FullyShardedNativeMixedPrecisionPlugin``
     - use native FSDP instead
     - #16400 :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.plugins.precision.sharded_native_amp.ShardedNativeMixedPrecisionPlugin``
     - use native FSDP instead
     - #16400 :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.strategies.fully_sharded.DDPFullyShardedStrategy``
     - use native FSDP instead
     - #16400 :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.strategies.sharded.DDPShardedStrategy``
     - use native FSDP instead
     - #16400 :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.strategies.sharded_spawn.DDPSpawnShardedStrategy``
     - use native FSDP instead
     - #16400 :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``save_config_overwrite`` parameters in ``LightningCLI``
     - pass this option and via dictionary of ``save_config_kwargs`` parameter
     - #14998

   * - used ``save_config_multifile`` parameters in ``LightningCLI``
     - pass this option and via dictionary of ``save_config_kwargs`` parameter
     - #14998

   * - have customized loops ``Loop.replace()``
     - implement your training loop with Fabric.
     - #14998 `Fabric`_

   * - have customized loops ``Loop.run()``
     - implement your training loop with Fabric.
     - #14998 `Fabric`_

   * - have customized loops ``Loop.connect()``
     - implement your training loop with Fabric.
     - #14998 `Fabric`_

   * - used the Trainer’s ``trainer.fit_loop`` property
     - implement your training loop with Fabric
     - #14998 `Fabric`_

   * - used the Trainer’s ``trainer.validate_loop`` property
     - implement your training loop with Fabric
     - #14998 `Fabric`_

   * - used the Trainer’s ``trainer.test_loop`` property
     - implement your training loop with Fabric
     - #14998 `Fabric`_

   * - used the Trainer’s ``trainer.predict_loop`` property
     - implement your training loop with Fabric
     - #14998 `Fabric`_

   * - used the ``Trainer.loop`` and fetching classes
     - being marked as protected
     -

   * - used ``opt_idx`` argument in ``BaseFinetuning.finetune_function``
     - use manual optimization
     - #16539

   * - used ``opt_idx`` argument in ``Callback.on_before_optimizer_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` as an optional argument in ``LightningModule.training_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.on_before_optimizer_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.configure_gradient_clipping``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.optimizer_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.optimizer_zero_grad``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.lr_scheduler_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used declaring optimizer frequencies in the dictionary returned from ``LightningModule.configure_optimizers``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer`` argument in ``LightningModule.backward``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.backward``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``PrecisionPlugin.optimizer_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``PrecisionPlugin.,backward``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``PrecisionPlugin.optimizer_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``Strategy.backward``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``Strategy.optimizer_step``
     - use manual optimization
     - #16539 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used Trainer’s ``Trainer.optimizer_frequencies`` attribute
     - use manual optimization
     - :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``PL_INTER_BATCH_PARALLELISM`` environment flag
     -
     - #16355

   * - used training integration with Horovod
     - install standalone package/project
     - https://github.com/Lightning-AI/lightning-Horovod

   * - used training integration with ColossalAI
     - install standalone package/project
     - https://lightning.ai/docs/pytorch/latest/advanced/third_party/colossalai.html

   * - used ``QuantizationAwareTraining`` callback
     - use Torch’s Quantization directly
     - #16750

   * - had any logic except reducing the DP outputs in  ``LightningModule.training_step_end`` hook
     - port it to ``LightningModule.training_batch_end`` hook
     - #16791

   * - had any logic except reducing the DP outputs in  ``LightningModule.validation_step_end`` hook
     - port it to ``LightningModule.validation_batch_end`` hook
     - #16791

   * - had any logic except reducing the DP outputs in  ``LightningModule.test_step_end`` hook
     - port it to ``LightningModule.test_batch_end`` hook
     - #16791

   * - used ``pl.strategies.DDPSpawnStrategy``
     - switch to general  ``DDPStrategy(start_method='spawn')`` with proper starting method
     - #16809

   * - used the automatic addition of a moving average of the ``training_step`` loss in the progress bar
     - use ``self.log("loss", ..., prog_bar=True)`` instead.
     - #16192

   * - rely on the ``outputs`` argument from the ``on_predict_epoch_end`` hook
     - access them via ``trainer.predict_loop.predictions``
     - #16655

   * - need to pass a dictionary to ``self.log()``
     - pass them independently.
     - #16389


.. _Fabric: https://lightning.ai/docs/fabric/
