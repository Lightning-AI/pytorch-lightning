.. list-table:: devel 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - passed the ``pl_module`` argument to distributed module wrappers
     - passed the (required) ``forward_module`` argument
     - #16386

   * - used the pl.plugins.ApexMixedPrecisionPlugin`` plugin
     - use PyTorch native mixed precision
     - #16039

   * - used the ``pl.plugins.NativeMixedPrecisionPlugin`` plugin
     - switch to the pl.plugins.MixedPrecisionPlugin`` plugin
     - #16039

   * - used the fit_loop.{min,max}_steps`` setters
     - implement your training loop with Fabric
     - #16803

   * - used the ``data_parallel`` attribute in ``Trainer``
     - check the same using ``isinstance(trainer.strategy, ParallelStrategy)`` instead
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
     - it was removed
     - #16424

   * - used any code from ``pl.utilities.distributed``
     - it was removed
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
     - #16172

   * - were using truncated backpropagation through time (TBPTT) with ``LightningModule.tbptt_split_batch``
     - use manual optimization
     - #16172

   * - were using truncated backpropagation through time (TBPTT) and passing ``hidden``  to ``LightningModule.training_step``
     - use manual optimization
     - #16172

   * - used ``pl.utilities.finite_checks.print_nan_gradients``
     - it was removed
     - ...

   * - used ``pl.utilities.finite_checks.detect_nan_parameters``
     - it was removed
     - ...

   * - used ``pl.utilities.parsing.flatten_dict``
     - it was removed
     - ...

   * - used ``pl.utilities.metrics.metrics_to_scalars``
     - it was removed
     - ...

   * - used ``pl.utilities.memory.get_model_size_mb``
     - it was removed
     - ...

   * - used pl.strategies.utils.on_colab_kaggle`` function
     - it was removed
     - #16437

   * - used ``add_argparse_args()`` from class ``LightningDataModule``
     - switch to using ``LightningCLI``
     - #16708

   * - used ``parse_argparser()`` from class ``LightningDataModule``
     - switch to using ``LightningCLI``
     - #16708

   * - used ``from_argparse_args()`` from class ``LightningDataModule``
     - switch to using ``LightningCLI``
     - #16708

   * - used ``get_init_arguments_and_types()`` from class ``LightningDataModule``
     - switch to using ``LightningCLI``
     - #16708

   * - used Trainer’s method ``default_attributes()``
     - switch to using ``LightningCLI``
     - #16708

   * - used Trainer’s method ``from_argparse_args()``
     - switch to using ``LightningCLI``
     - #16708

   * - used Trainer’s method ``parse_argparser()``
     - switch to using ``LightningCLI``
     - #16708

   * - used Trainer’s method ``match_env_arguments()``
     - switch to using ``LightningCLI``
     - #16708

   * - used Trainer’s method ``add_argparse_args()``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``from_argparse_args()`` from pl.utilities.argparse``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``parse_argparser()`` from pl.utilities.argparse``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``parse_env_variables()`` from pl.utilities.argparse``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``get_init_arguments_and_types()`` from pl.utilities.argparse``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``add_argparse_args()`` from pl.utilities.argparse``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``str_to_bool()`` from pl.utilities.parsing``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``str_to_bool_or_int()`` from pl.utilities.parsing``
     - switch to using ``LightningCLI``
     - #16708

   * - used function ``str_to_bool_or_str()`` from pl.utilities.parsing``
     - switch to using ``LightningCLI``
     - #16708

   * - used ``pl.utilities.distributed.AllGatherGrad``
     - switch to PyTorch native one
     - #15364

   * - used ``PL_RECONCILE_PROCESS=1``
     - it does not have any effect
     - #16204

   * - Mixin’s method ``load_from_checkpoint`` was moved from ``pl.core.saving.ModelIO`` to ``pl.core.module.LightningModule``
     - ...
     - #16999

   * - used  Accelerator.setup_environment`` method
     - switch to ``Accelerator.setup_device``  instead
     - #16436

   * - used PL_FAULT_TOLERANT_TRAINING`` environment flag
     - it does not have any effect
     - #16516 #16533

   * - used or derived from public ``pl.overrides.distributed.IndexBatchSamplerWrapper`` class
     - it is set as protected
     - #16826

   * - used the DataLoaderLoop`` class
     - use manual optimization
     - #16726

   * - used the EvaluationEpochLoop`` class
     - use manual optimization
     - #16726

   * - used the PredictionEpochLoop`` class
     - use manual optimization
     - #16726

   * - used ``trainer.reset_*_dataloader()`` methods
     - use  Loop.setup_data()`` for the top-level loops
     - #16726

   * - used LightningModule.precision`` attribute
     - rely on Trainer precison
     - #16203

   * - used  Trainer.model`` setter
     - you shall pass the ``model`` in fit/test/predict method
     - #16462

   * - relied on pl.utilities.supporters.CombinedLoaderIterator`` class
     - pass dataloders directly
     - #16714

   * - relied on pl.utilities.supporters.CombinedLoaderIterator`` class
     - pass dataloders directly
     - #16714

   * - accessed  ProgressBarBase.train_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - accessed  ProgressBarBase.val_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - accessed  ProgressBarBase.test_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - accessed  ProgressBarBase.predict_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - #16760

   * - used Trainer.prediction_writer_callbacks`` property
     - rely on precision plugin
     - #16759

   * - used PrecisionPlugin.dispatch``
     - it was removed
     - #16618

   * - used Strategy.dispatch``
     - it was removed
     - #16618
