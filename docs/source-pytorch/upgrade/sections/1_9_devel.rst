.. list-table:: devel 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - passed the ``pl_module`` argument to distributed module wrappers
     - passed the (required) ``forward_module`` argument
     - #16386

   * - used ``DataParallel`` and the ``LightningParallelModule`` wrapper
     - use DDP or DeepSpeed instead
     - #16748 :doc:`DDP <../../accelerators/gpu_expert>`

   * - used ``pl_module`` argument from the distributed module wrappers
     - use DDP or DeepSpeed instead
     - #16386 :doc:`DDP <../../accelerators/gpu_expert>`

   * - called ``pl.overrides.base.unwrap_lightning_module`` function
     - use DDP or DeepSpeed instead
     - #16386 :doc:`DDP <../../accelerators/gpu_expert>`

   * - used or derived from ``pl.overrides.distributed.LightningDistributedModule`` class
     - use DDP instead
     - #16386 :doc:`DDP <../../accelerators/gpu_expert>`

   * - used the pl.plugins.ApexMixedPrecisionPlugin`` plugin
     - use PyTorch native mixed precision
     - #16039

   * - used the ``pl.plugins.NativeMixedPrecisionPlugin`` plugin
     - switch to the ``pl.plugins.MixedPrecisionPlugin`` plugin
     - #16039

   * - used the ``fit_loop.min_steps`` setters
     - implement your training loop with Fabric
     - #16803

   * - used the ``fit_loop.max_steps`` setters
     - implement your training loop with Fabric
     - #16803

   * - used the ``data_parallel`` attribute in ``Trainer``
     - check the same using ``isinstance(trainer.strategy, ParallelStrategy)``
     - #16703

   * - used any function from ``pl.utilities.xla_device``
     - switch to ``pl.accelerators.TPUAccelerator.is_available()``
     - #14514 #14550

   * - imported functions from  ``pl.utilities.device_parser.*``
     - import them from ``lightning_fabric.utilities.device_parser.*``
     - #14492 #14753

   * - imported functions from ``pl.utilities.cloud_io.*``
     - import them from ``lightning_fabric.utilities.cloud_io.*``
     - #14515

   * - imported functions from ``pl.utilities.apply_func.*``
     - import them from ``lightning_utilities.core.apply_func.*``
     - #14516 #14537

   * - used any code from ``pl.core.mixins``
     - use the base classes
     - #16424

   * - used any code from ``pl.utilities.distributed``
     - rely on Pytorch's native functions
     - #16390

   * - used any code from ``pl.utilities.data``
     - it was removed
     - #16440

   * - used any code from ``pl.utilities.optimizer``
     - it was removed
     - #16439

   * - used any code from ``pl.utilities.seed``
     - it was removed
     - #16422

   * - were using truncated backpropagation through time (TBPTT) with ``LightningModule.truncated_bptt_steps``
     - use manual optimization
     - #16172 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - were using truncated backpropagation through time (TBPTT) with ``LightningModule.tbptt_split_batch``
     - use manual optimization
     - #16172 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - were using truncated backpropagation through time (TBPTT) and passing ``hidden``  to ``LightningModule.training_step``
     - use manual optimization
     - #16172 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``pl.utilities.finite_checks.print_nan_gradients`` function
     - it was removed
     - ...

   * - used ``pl.utilities.finite_checks.detect_nan_parameters`` function
     - it was removed
     - ...

   * - used ``pl.utilities.parsing.flatten_dict`` function
     - it was removed
     - ...

   * - used ``pl.utilities.metrics.metrics_to_scalars`` function
     - it was removed
     - ...

   * - used ``pl.utilities.memory.get_model_size_mb`` function
     - it was removed
     - ...

   * - used ``pl.strategies.utils.on_colab_kaggle`` function
     - it was removed
     - #16437

   * - used ``LightningDataModule.add_argparse_args()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``LightningDataModule.parse_argparser()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``LightningDataModule.from_argparse_args()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``LightningDataModule.get_init_arguments_and_types()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``Trainer.default_attributes()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``Trainer.from_argparse_args()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``Trainer.parse_argparser()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``Trainer.match_env_arguments()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``Trainer.add_argparse_args()`` method
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.argparse.from_argparse_args()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.argparse.parse_argparser()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.argparseparse_env_variables()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``get_init_arguments_and_types()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.argparse.add_argparse_args()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.parsing.str_to_bool()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.parsing.str_to_bool_or_int()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.parsing.str_to_bool_or_str()`` function
     - switch to using ``LightningCLI``
     - #16708

   * - derived from ``pl.utilities.distributed.AllGatherGrad`` class
     - switch to PyTorch native equivalent
     - #15364

   * - used ``PL_RECONCILE_PROCESS=1`` env. variable
     - customize your logger
     - #16204

   * - if you derived from mixin’s method ``pl.core.saving.ModelIO.load_from_checkpoint``
     - rely on ``pl.core.module.LightningModule``
     - #16999

   * - used  ``Accelerator.setup_environment`` method
     - switch to ``Accelerator.setup_device``
     - #16436

   * - used ``PL_FAULT_TOLERANT_TRAINING`` env. variable
     - implement own logic with Fabric
     - #16516 #16533

   * - used or derived from public ``pl.overrides.distributed.IndexBatchSamplerWrapper`` class
     - it is set as protected
     - #16826

   * - used the ``DataLoaderLoop`` class
     - use manual optimization
     - #16726 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used the ``EvaluationEpochLoop`` class
     - use manual optimization
     - #16726 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used the ``PredictionEpochLoop`` class
     - use manual optimization
     - #16726 :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``trainer.reset_*_dataloader()`` methods
     - use  ``Loop.setup_data()`` for the top-level loops
     - #16726

   * - used ``LightningModule.precision`` attribute
     - rely on Trainer precision attribute
     - #16203

   * - used  ``Trainer.model`` setter
     - you shall pass the ``model`` in fit/test/predict method
     - #16462

   * - relied on ``pl.utilities.supporters.CombinedLoaderIterator`` class
     - pass dataloders directly
     - #16714

   * - relied on ``pl.utilities.supporters.CombinedLoaderIterator`` class
     - pass dataloders directly
     - #16714

   * - accessed ``ProgressBarBase.train_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - accessed ``ProgressBarBase.val_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - accessed ``ProgressBarBase.test_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - accessed ``ProgressBarBase.predict_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - used ``Trainer.prediction_writer_callbacks`` property
     - rely on precision plugin
     - #16759

   * - used ``PrecisionPlugin.dispatch``
     - it was removed
     - #16618

   * - used ``Strategy.dispatch``
     - it was removed
     - #16618
