# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.5.3] - 2019-11-06
### Added
- Added option to disable default logger, checkpointer, and early stopping by passing `logger=False`, `checkpoint_callback=False` and `early_stop_callback=False` respectively
- Added `CometLogger` for use with Comet.ml
- Added `val_check_interval` argument to `Trainer` allowing validition to be performed at every given number of batches
- Added functionality to save and load hyperparameters using the standard checkpoint mechanism
- Added call to `torch.cuda.empty_cache` before training starts
- Added option for user to override the call t `backward`
- Added support for truncated backprop through time via the `truncated_bptt_steps` argument in `Trainer` 
- Added option to operate on all outputs from `training_step` in DDP2
- Added a hook for modifying DDP init
- Added a hook for modifying Apex
### Changed
- Changed experiment version to be padded with zeros (e.g. `/dir/version_9` becomes `/dir/version_0009`)
- Changed callback metrics to include any metrics given in logs or progress bar
- Changed the default for `save_best_only` in `ModelCheckpoint` to `True`
- Added `tng_data_loader` for backwards compatibility
- Renamed `MLFlowLogger.client` to `MLFlowLogger.experiment` for consistency
- Moved `global_step` increment to happen after the batch has been processed
- Changed weights restore to first attempt HPC weights before restoring normally, preventing both weights being restored and running out of memory
- Changed progress bar functionality to add multiple progress bars for train/val/test
- Changed calls to `print` to use `logging` instead
### Deprecated
- Deprecated `tng_dataloader`
### Removed
### Fixed
- Fixed an issue where the number of batches was off by one during training
- Fixed a bug that occured when setting a ckeckpoint callback and `early_stop_callback=False`
- Fixed an error when importing CometLogger
- Fixed a bug where the `gpus` argument had some unexpected behaviour
- Fixed a bug where the computed total number of batches was sometimes incorrect
- Fixed a bug where the progress bar would sometimes not show the total number of batches in test mode
- Fixed a bug when using the `log_gpu_memory='min_max'` option in `Trainer`
- Fixed a bug where checkpointing would sometimes erase the current directory

## [0.5.2] - 2019-10-10
### Added
- Added `weights_summary` argument to `Trainer` to be set to `full` (full summary), `top` (just top level modules) or other
- Added `tags` argument to `MLFlowLogger`
### Changed
- Changed default for `amp_level` to `O1`
### Deprecated
### Removed
- Removed the `print_weights_summary` argument from `Trainer`
### Fixed
- Fixed a bug where logs were not written properly
- Fixed a bug where `logger.finalize` wasn't called after training is complete
- Fixed callback metric errors in DDP
- Fixed a bug where `TestTubeLogger` didn't log to the correct directory

## [0.5.1] - 2019-10-05
### Added
- Added the `LightningLoggerBase` class for experiment loggers
- Added `MLFlowLogger` for logging with `mlflow`
- Added `TestTubeLogger` for logging with `test_tube`
- Added a different implementation of DDP (`distributed_backed='ddp2'`) where every node has one model using all GPUs
- Added support for optimisers which require a closure (e.g. LBFGS)
- Added automatic `MASTER_PORT` defualt for DDP when not set manually
- Added new GPU memory logging options `'min_max'` (log only the min/max utilization) and `'all'` (log all the GPU memory)
### Changed
- Changed schedulers to always be called with the current epoch
- Changed `test_tube` to an optional dependency
- Changed data loaders to internally use a getter instead of a python property
- Disabled auto GPU loading when restoring weights to prevent out of memory errors
- Changed logging, early stopping and checkpointing to occur by default
### Deprecated
### Removed
### Fixed
- Fixed a bug with samplers that do not specify `set_epoch`
- Fixed a bug when using the `MLFlowLogger` with unsupported data types, this will now raise a warning
- Fixed a bug where gradient norms were alwasy zero using `track_grad_norm`
- Fixed a bug which causes a crash when logging memory

## [0.5.0] - 2019-09-26
### Added
### Changed
- Changed `data_batch` argument to `batch` throughout
- Changed `batch_i` argument to `batch_idx` throughout
- Changed `tng_dataloader` method to `train_dataloader`
- Changed `on_tng_metrics` method to `on_training_metrics`
- Changed `gradient_clip` argument to `gradient_clip_val`
- Changed `add_log_row_interval` to `row_log_interval`
### Deprecated
### Removed
### Fixed
- Fixed a bug with tensorboard logging in multi-gpu setup

## [0.4.9] - 2019-09-16
### Added
- Added the flag `log_gpu_memory` to `Trainer` to deactivate logging of GPU
memory utilization
- Added SLURM resubmit functionality (port from test-tube)
- Added optional weight_save_path to trainer to remove the need for a checkpoint_callback when using cluster training
- Added option to use single gpu per node with `DistributedDataParrallel`
### Changed
- Changed functionality of `validation_end` and `test_end` with multiple dataloaders to be given all of the dataloaders at once rather than in seperate calls
- Changed print_nan_grads to only print the parameter value and gradients when they contain NaN
- Changed gpu API to take integers as well (e.g. `gpus=2` instead of `gpus=[0, 1]`)
- All models now loaded on to CPU to avoid device and out of memory issues in PyTorch
### Deprecated
### Removed
### Fixed
- Fixed a bug where data types that implement `.to` but not `.cuda` would not be properly moved onto the GPU
- Fixed a bug where data would not be re-shuffled every epoch when using a `DistributedSampler`

## [0.4.8] - 2019-08-31
### Added
- Added `test_step` and `test_end` methods, used when `Trainer.test` is called
- Added `GradientAccumulationScheduler` callback which can be used to schedule changes to the number of accumulation batches
- Added option to skip the validation sanity check by setting `nb_sanity_val_steps = 0`
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug when setting `nb_sanity_val_steps = 0`

## [0.4.7] - 2019-08-24
### Added
### Changed
- Changed the default `val_check_interval` to `1.0`
- Changed defaults for `nb_val_batches`, `nb_tng_batches` and `nb_test_batches` to 0
### Deprecated
### Removed
### Fixed
- Fixed a bug where the full validation set as used despite setting `val_percent_check`
- Fixed a bug where an `Exception` was thrown when using a data set containing a single batch
- Fixed a bug where an `Exception` was thrown if no `val_dataloader` was given
- Fixed a bug where tuples were not properly transfered to the GPU
- Fixed a bug where data of a non standard type was not properly handled by the trainer
- Fixed a bug when loading data as a tuple
- Fixed a bug where `AttributeError` could be suppressed by the `Trainer`

## [0.4.6] - 2019-08-15
### Added
- Added support for data to be given as a `dict` or `list` with a single gpu
- Added support for `configure_optimizers` to return a single optimizer, two list (optimizers and schedulers), or a single list
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where returning just an optimizer list (i.e. without schedulers) from `configure_optimizers` would throw an `Exception`

## [0.4.5] - 2019-08-13
### Added
- Added `optimizer_step` method that can be overridden to change the standard optimizer behaviour
### Changed
### Deprecated
### Removed
### Fixed

## [0.4.4] - 2019-08-12
### Added
- Added supoort for multiple validation dataloaders
- Added support for latest test-tube logger (optimised for `torch==1.2.0`)
### Changed
- `validation_step` and `val_dataloader` are now optional
- `lr_scheduler` is now activated after epoch
### Deprecated
### Removed
### Fixed
- Fixed a bug where a warning would show when using `lr_scheduler` in `torch>1.1.0`
- Fixed a bug where an `Exception` would be thrown if using `torch.DistributedDataParrallel` without using a `DistributedSampler`, this now throws a `Warning` instead 

## [0.4.3] - 2019-08-10
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where accumulate gradients would scale the loss incorrectly

## [0.4.2] - 2019-08-08
### Added
### Changed
- Changed install requirement to `torch==1.2.0`
### Deprecated
### Removed
### Fixed

## [0.4.1] - 2019-08-08
### Added
### Changed
- Changed install requirement to `torch==1.1.0`
### Deprecated
### Removed
### Fixed

## [0.4.0] - 2019-08-08
### Added
- Added 16-bit support for a single GPU
- Added support for training continuation (preserves epoch, global step etc.)
### Changed
- Changed `training_step` and `validation_step`, outputs will no longer be automatically reduced
### Deprecated
### Removed
- Removed need for `Experiment` object in `Trainer`
### Fixed
- Fixed issues with reducing outputs from generative models (such as images and text)

## [0.3.6.1] - 2019-07-27
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Fixed a bug where `Experiment` object was not process safe, potentially causing logs to be overwritten

## [0.3.6] - 2019-07-25
### Added
- Added a decorator to do lazy data loading internally
### Changed
### Deprecated
### Removed
### Fixed
