.. list-table:: adv. user 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used the ``pl.lite`` module
     - switch to ``lightning_fabric``
     - `PR15953`_

   * - used Trainer’s flag ``strategy='dp'``
     - use DDP with ``strategy='ddp'`` or DeepSpeed instead
     - `PR16748`_

   * - implemented ``LightningModule.training_epoch_end`` hooks
     - port your logic to  ``LightningModule.on_train_epoch_end`` hook
     - `PR16520`_

   * - implemented ``LightningModule.validation_epoch_end`` hook
     - port your logic to  ``LightningModule.on_validation_epoch_end`` hook
     - `PR16520`_

   * - implemented ``LightningModule.test_epoch_end`` hooks
     - port your logic to  ``LightningModule.on_test_epoch_end`` hook
     - `PR16520`_

   * - used Trainer’s flag ``multiple_trainloader_mode``
     - switch to  ``CombinedLoader(..., mode=...)`` and set mode directly now
     - `PR16800`_

   * - used Trainer’s flag ``move_metrics_to_cpu``
     - implement particular offload logic in your custom metric or turn it on in ``torchmetrics``
     - `PR16358`_

   * - used Trainer’s flag ``track_grad_norm``
     - overwrite ``on_before_optimizer_step`` hook and pass the argument directly and ``LightningModule.log_grad_norm()`` hook
     - `PR16745`_ `PR16745`_

   * - used Trainer’s flag ``replace_sampler_ddp``
     - use  ``use_distributed_sampler``; the sampler gets created not only for the DDP strategies
     -

   * - relied on the ``on_tpu`` argument in ``LightningModule.optimizer_step`` hook
     - switch to manual optimization
     - `PR16537`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - relied on the ``using_lbfgs`` argument in ``LightningModule.optimizer_step`` hook
     - switch to manual optimization
     - `PR16538`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - were using ``nvidia/apex`` in any form
     - switch to PyTorch native mixed precision ``torch.amp`` instead
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - used Trainer’s flag ``using_native_amp``
     - use PyTorch native mixed precision
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - used Trainer’s flag ``amp_backend``
     - use PyTorch native mixed precision
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - used Trainer’s flag ``amp_level``
     - use PyTorch native mixed precision
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - used Trainer’s attribute ``using_native_amp``
     - use PyTorch native mixed precision
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - used Trainer’s attribute ``amp_backend``
     - use PyTorch native mixed precision
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - used Trainer’s attribute ``amp_level``
     - use PyTorch native mixed precision
     - `PR16039`_ :doc:`Precision <../../common/precision>`

   * - use the ``FairScale`` integration
     - consider using PyTorch's native FSDP implementation or outsourced implementation into own project
     - `lightning-Fairscale`_

   * - used ``pl.overrides.fairscale.LightningShardedDataParallel``
     - use native FSDP instead
     - `PR16400`_ :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.plugins.precision.fully_sharded_native_amp.FullyShardedNativeMixedPrecisionPlugin``
     - use native FSDP instead
     - `PR16400`_ :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.plugins.precision.sharded_native_amp.ShardedNativeMixedPrecisionPlugin``
     - use native FSDP instead
     - `PR16400`_ :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.strategies.fully_sharded.DDPFullyShardedStrategy``
     - use native FSDP instead
     - `PR16400`_ :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.strategies.sharded.DDPShardedStrategy``
     - use native FSDP instead
     - `PR16400`_ :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``pl.strategies.sharded_spawn.DDPSpawnShardedStrategy``
     - use native FSDP instead
     - `PR16400`_ :doc:`FSDP <../../accelerators/gpu_expert>`

   * - used ``save_config_overwrite`` parameters in ``LightningCLI``
     - pass this option and via dictionary of ``save_config_kwargs`` parameter
     - `PR14998`_

   * - used ``save_config_multifile`` parameters in ``LightningCLI``
     - pass this option and via dictionary of ``save_config_kwargs`` parameter
     - `PR14998`_

   * - have customized loops ``Loop.replace()``
     - implement your training loop with Fabric.
     - `PR14998`_ `Fabric`_

   * - have customized loops ``Loop.run()``
     - implement your training loop with Fabric.
     - `PR14998`_ `Fabric`_

   * - have customized loops ``Loop.connect()``
     - implement your training loop with Fabric.
     - `PR14998`_ `Fabric`_

   * - used the Trainer’s ``trainer.fit_loop`` property
     - implement your training loop with Fabric
     - `PR14998`_ `Fabric`_

   * - used the Trainer’s ``trainer.validate_loop`` property
     - implement your training loop with Fabric
     - `PR14998`_ `Fabric`_

   * - used the Trainer’s ``trainer.test_loop`` property
     - implement your training loop with Fabric
     - `PR14998`_ `Fabric`_

   * - used the Trainer’s ``trainer.predict_loop`` property
     - implement your training loop with Fabric
     - `PR14998`_ `Fabric`_

   * - used the ``Trainer.loop`` and fetching classes
     - being marked as protected
     -

   * - used ``opt_idx`` argument in ``BaseFinetuning.finetune_function``
     - use manual optimization
     - `PR16539`_

   * - used ``opt_idx`` argument in ``Callback.on_before_optimizer_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` as an optional argument in ``LightningModule.training_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.on_before_optimizer_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.configure_gradient_clipping``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.optimizer_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.optimizer_zero_grad``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.lr_scheduler_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used declaring optimizer frequencies in the dictionary returned from ``LightningModule.configure_optimizers``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer`` argument in ``LightningModule.backward``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``LightningModule.backward``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``PrecisionPlugin.optimizer_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``PrecisionPlugin.,backward``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``PrecisionPlugin.optimizer_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``Strategy.backward``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``optimizer_idx`` argument in ``Strategy.optimizer_step``
     - use manual optimization
     - `PR16539`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used Trainer’s ``Trainer.optimizer_frequencies`` attribute
     - use manual optimization
     - :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``PL_INTER_BATCH_PARALLELISM`` environment flag
     -
     - `PR16355`_

   * - used training integration with Horovod
     - install standalone package/project
     - `lightning-Horovod`_

   * - used training integration with ColossalAI
     - install standalone package/project
     - `lightning-ColossalAI`_

   * - used ``QuantizationAwareTraining`` callback
     - use Torch’s Quantization directly
     - `PR16750`_

   * - had any logic except reducing the DP outputs in  ``LightningModule.training_step_end`` hook
     - port it to ``LightningModule.on_train_batch_end`` hook
     - `PR16791`_

   * - had any logic except reducing the DP outputs in  ``LightningModule.validation_step_end`` hook
     - port it to ``LightningModule.on_validation_batch_end`` hook
     - `PR16791`_

   * - had any logic except reducing the DP outputs in  ``LightningModule.test_step_end`` hook
     - port it to ``LightningModule.on_test_batch_end`` hook
     - `PR16791`_

   * - used ``pl.strategies.DDPSpawnStrategy``
     - switch to general  ``DDPStrategy(start_method='spawn')`` with proper starting method
     - `PR16809`_

   * - used the automatic addition of a moving average of the ``training_step`` loss in the progress bar
     - use ``self.log("loss", ..., prog_bar=True)`` instead.
     - `PR16192`_

   * - rely on the ``outputs`` argument from the ``on_predict_epoch_end`` hook
     - access them via ``trainer.predict_loop.predictions``
     - `PR16655`_

   * - need to pass a dictionary to ``self.log()``
     - pass them independently.
     - `PR16389`_


.. _Fabric: https://lightning.ai/docs/fabric/
.. _lightning-Horovod: https://github.com/Lightning-AI/lightning-Horovod
.. _lightning-ColossalAI: https://lightning.ai/docs/pytorch/2.1.0/integrations/strategies/colossalai.html
.. _lightning-Fairscale: https://github.com/Lightning-Sandbox/lightning-Fairscale

.. _pr15953: https://github.com/Lightning-AI/lightning/pull/15953
.. _pr16748: https://github.com/Lightning-AI/lightning/pull/16748
.. _pr16520: https://github.com/Lightning-AI/lightning/pull/16520
.. _pr16800: https://github.com/Lightning-AI/lightning/pull/16800
.. _pr16358: https://github.com/Lightning-AI/lightning/pull/16358
.. _pr16745: https://github.com/Lightning-AI/lightning/pull/16745
.. _pr16537: https://github.com/Lightning-AI/lightning/pull/16537
.. _pr16538: https://github.com/Lightning-AI/lightning/pull/16538
.. _pr16039: https://github.com/Lightning-AI/lightning/pull/16039
.. _pr16400: https://github.com/Lightning-AI/lightning/pull/16400
.. _pr14998: https://github.com/Lightning-AI/lightning/pull/14998
.. _pr16539: https://github.com/Lightning-AI/lightning/pull/16539
.. _pr16355: https://github.com/Lightning-AI/lightning/pull/16355
.. _pr16750: https://github.com/Lightning-AI/lightning/pull/16750
.. _pr16791: https://github.com/Lightning-AI/lightning/pull/16791
.. _pr16809: https://github.com/Lightning-AI/lightning/pull/16809
.. _pr16192: https://github.com/Lightning-AI/lightning/pull/16192
.. _pr16655: https://github.com/Lightning-AI/lightning/pull/16655
.. _pr16389: https://github.com/Lightning-AI/lightning/pull/16389
