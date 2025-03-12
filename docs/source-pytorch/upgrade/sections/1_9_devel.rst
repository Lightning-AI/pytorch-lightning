.. list-table:: devel 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - passed the ``pl_module`` argument to distributed module wrappers
     - passed the (required) ``forward_module`` argument
     - `PR16386`_

   * - used ``DataParallel`` and the ``LightningParallelModule`` wrapper
     - use DDP or DeepSpeed instead
     - `PR16748`_ :doc:`DDP <../../accelerators/gpu_expert>`

   * - used ``pl_module`` argument from the distributed module wrappers
     - use DDP or DeepSpeed instead
     - `PR16386`_ :doc:`DDP <../../accelerators/gpu_expert>`

   * - called ``pl.overrides.base.unwrap_lightning_module`` function
     - use DDP or DeepSpeed instead
     - `PR16386`_ :doc:`DDP <../../accelerators/gpu_expert>`

   * - used or derived from ``pl.overrides.distributed.LightningDistributedModule`` class
     - use DDP instead
     - `PR16386`_ :doc:`DDP <../../accelerators/gpu_expert>`

   * - used the ``pl.plugins.ApexMixedPrecisionPlugin`` plugin
     - use PyTorch native mixed precision
     - `PR16039`_

   * - used the ``pl.plugins.NativeMixedPrecisionPlugin`` plugin
     - switch to the ``pl.plugins.MixedPrecisionPlugin`` plugin
     - `PR16039`_

   * - used the ``fit_loop.min_steps`` setters
     - implement your training loop with Fabric
     - `PR16803`_

   * - used the ``fit_loop.max_steps`` setters
     - implement your training loop with Fabric
     - `PR16803`_

   * - used the ``data_parallel`` attribute in ``Trainer``
     - check the same using ``isinstance(trainer.strategy, ParallelStrategy)``
     - `PR16703`_

   * - used any function from ``pl.utilities.xla_device``
     - switch to ``pl.accelerators.XLAAccelerator.is_available()``
     - `PR14514`_ `PR14550`_

   * - imported functions from  ``pl.utilities.device_parser.*``
     - import them from ``lightning_fabric.utilities.device_parser.*``
     - `PR14492`_ `PR14753`_

   * - imported functions from ``pl.utilities.cloud_io.*``
     - import them from ``lightning_fabric.utilities.cloud_io.*``
     - `PR14515`_

   * - imported functions from ``pl.utilities.apply_func.*``
     - import them from ``lightning_utilities.core.apply_func.*``
     - `PR14516`_ `PR14537`_

   * - used any code from ``pl.core.mixins``
     - use the base classes
     - `PR16424`_

   * - used any code from ``pl.utilities.distributed``
     - rely on Pytorch's native functions
     - `PR16390`_

   * - used any code from ``pl.utilities.data``
     - it was removed
     - `PR16440`_

   * - used any code from ``pl.utilities.optimizer``
     - it was removed
     - `PR16439`_

   * - used any code from ``pl.utilities.seed``
     - it was removed
     - `PR16422`_

   * - were using truncated backpropagation through time (TBPTT) with ``LightningModule.truncated_bptt_steps``
     - use manual optimization
     - `PR16172`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - were using truncated backpropagation through time (TBPTT) with ``LightningModule.tbptt_split_batch``
     - use manual optimization
     - `PR16172`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - were using truncated backpropagation through time (TBPTT) and passing ``hidden``  to ``LightningModule.training_step``
     - use manual optimization
     - `PR16172`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``pl.utilities.finite_checks.print_nan_gradients`` function
     - it was removed
     -

   * - used ``pl.utilities.finite_checks.detect_nan_parameters`` function
     - it was removed
     -

   * - used ``pl.utilities.parsing.flatten_dict`` function
     - it was removed
     -

   * - used ``pl.utilities.metrics.metrics_to_scalars`` function
     - it was removed
     -

   * - used ``pl.utilities.memory.get_model_size_mb`` function
     - it was removed
     -

   * - used ``pl.strategies.utils.on_colab_kaggle`` function
     - it was removed
     - `PR16437`_

   * - used ``LightningDataModule.add_argparse_args()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``LightningDataModule.parse_argparser()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``LightningDataModule.from_argparse_args()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``LightningDataModule.get_init_arguments_and_types()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``Trainer.default_attributes()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``Trainer.from_argparse_args()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``Trainer.parse_argparser()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``Trainer.match_env_arguments()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``Trainer.add_argparse_args()`` method
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.argparse.from_argparse_args()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.argparse.parse_argparser()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.argparseparse_env_variables()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``get_init_arguments_and_types()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.argparse.add_argparse_args()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.parsing.str_to_bool()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.parsing.str_to_bool_or_int()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - used ``pl.utilities.parsing.str_to_bool_or_str()`` function
     - switch to using ``LightningCLI``
     - `PR16708`_

   * - derived from ``pl.utilities.distributed.AllGatherGrad`` class
     - switch to PyTorch native equivalent
     - `PR15364`_

   * - used ``PL_RECONCILE_PROCESS=1`` env. variable
     - customize your logger
     - `PR16204`_

   * - if you derived from mixin’s method ``pl.core.saving.ModelIO.load_from_checkpoint``
     - rely on ``pl.core.module.LightningModule``
     - `PR16999`_

   * - used  ``Accelerator.setup_environment`` method
     - switch to ``Accelerator.setup_device``
     - `PR16436`_

   * - used ``PL_FAULT_TOLERANT_TRAINING`` env. variable
     - implement own logic with Fabric
     - `PR16516`_ `PR16533`_

   * - used or derived from public ``pl.overrides.distributed.IndexBatchSamplerWrapper`` class
     - it is set as protected
     - `PR16826`_

   * - used the ``DataLoaderLoop`` class
     - use manual optimization
     - `PR16726`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used the ``EvaluationEpochLoop`` class
     - use manual optimization
     - `PR16726`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used the ``PredictionEpochLoop`` class
     - use manual optimization
     - `PR16726`_ :doc:`Manual Optimization <../../model/manual_optimization>`

   * - used ``trainer.reset_*_dataloader()`` methods
     - use  ``Loop.setup_data()`` for the top-level loops
     - `PR16726`_

   * - used ``LightningModule.precision`` attribute
     - rely on Trainer precision attribute
     - `PR16203`_

   * - used  ``Trainer.model`` setter
     - you shall pass the ``model`` in fit/test/predict method
     - `PR16462`_

   * - relied on ``pl.utilities.supporters.CombinedLoaderIterator`` class
     - pass dataloders directly
     - `PR16714`_

   * - relied on ``pl.utilities.supporters.CombinedLoaderIterator`` class
     - pass dataloders directly
     - `PR16714`_

   * - used ``pl.callbacks.progress.base.ProgressBarBase``
     - rename to ``pl.callbacks.progress.ProgressBar``
     - `PR17058`_

   * - accessed ``ProgressBarBase.train_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - `PR16760`_

   * - accessed ``ProgressBarBase.val_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - `PR16760`_

   * - accessed ``ProgressBarBase.test_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - `PR16760`_

   * - accessed ``ProgressBarBase.predict_batch_idx`` property
     - rely on Trainer internal loops’ properties
     - `PR16760`_

   * - used ``Trainer.prediction_writer_callbacks`` property
     - rely on precision plugin
     - `PR16759`_

   * - used ``PrecisionPlugin.dispatch``
     - it was removed
     - `PR16618`_

   * - used ``Strategy.dispatch``
     - it was removed
     - `PR16618`_


.. _pr16386: https://github.com/Lightning-AI/lightning/pull/16386
.. _pr16748: https://github.com/Lightning-AI/lightning/pull/16748
.. _pr16039: https://github.com/Lightning-AI/lightning/pull/16039
.. _pr16803: https://github.com/Lightning-AI/lightning/pull/16803
.. _pr16703: https://github.com/Lightning-AI/lightning/pull/16703
.. _pr14514: https://github.com/Lightning-AI/lightning/pull/14514
.. _pr14550: https://github.com/Lightning-AI/lightning/pull/14550
.. _pr14492: https://github.com/Lightning-AI/lightning/pull/14492
.. _pr14753: https://github.com/Lightning-AI/lightning/pull/14753
.. _pr14515: https://github.com/Lightning-AI/lightning/pull/14515
.. _pr14516: https://github.com/Lightning-AI/lightning/pull/14516
.. _pr14537: https://github.com/Lightning-AI/lightning/pull/14537
.. _pr16424: https://github.com/Lightning-AI/lightning/pull/16424
.. _pr16390: https://github.com/Lightning-AI/lightning/pull/16390
.. _pr16440: https://github.com/Lightning-AI/lightning/pull/16440
.. _pr16439: https://github.com/Lightning-AI/lightning/pull/16439
.. _pr16422: https://github.com/Lightning-AI/lightning/pull/16422
.. _pr16172: https://github.com/Lightning-AI/lightning/pull/16172
.. _pr16437: https://github.com/Lightning-AI/lightning/pull/16437
.. _pr16708: https://github.com/Lightning-AI/lightning/pull/16708
.. _pr15364: https://github.com/Lightning-AI/lightning/pull/15364
.. _pr16204: https://github.com/Lightning-AI/lightning/pull/16204
.. _pr16999: https://github.com/Lightning-AI/lightning/pull/16999
.. _pr16436: https://github.com/Lightning-AI/lightning/pull/16436
.. _pr16516: https://github.com/Lightning-AI/lightning/pull/16516
.. _pr16533: https://github.com/Lightning-AI/lightning/pull/16533
.. _pr16826: https://github.com/Lightning-AI/lightning/pull/16826
.. _pr16726: https://github.com/Lightning-AI/lightning/pull/16726
.. _pr16203: https://github.com/Lightning-AI/lightning/pull/16203
.. _pr16462: https://github.com/Lightning-AI/lightning/pull/16462
.. _pr16714: https://github.com/Lightning-AI/lightning/pull/16714
.. _pr17058: https://github.com/Lightning-AI/lightning/pull/17058
.. _pr16760: https://github.com/Lightning-AI/lightning/pull/16760
.. _pr16759: https://github.com/Lightning-AI/lightning/pull/16759
.. _pr16618: https://github.com/Lightning-AI/lightning/pull/16618
