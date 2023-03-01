# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [1.9.4] - 2023-03-01

### Added

- Added `Fabric(strategy="auto")` support ([#16916](https://github.com/Lightning-AI/lightning/pull/16916))

### Fixed

- Fixed edge cases in parsing device ids using NVML ([#16795](https://github.com/Lightning-AI/lightning/pull/16795))
- Fixed DDP spawn hang on TPU Pods ([#16844](https://github.com/Lightning-AI/lightning/pull/16844))
- Fixed an error when passing `find_usable_cuda_devices(num_devices=-1)` ([#16866](https://github.com/Lightning-AI/lightning/pull/16866))


## [1.9.3] - 2023-02-21

### Fixed

- Fixed an issue causing a wrong environment plugin to be selected when `accelerator=tpu` and `devices > 1` ([#16806](https://github.com/Lightning-AI/lightning/pull/16806))
- Fixed parsing of defaults for `--accelerator` and `--precision` in Fabric CLI when `accelerator` and `precision` are set to non-default values in the code ([#16818](https://github.com/Lightning-AI/lightning/pull/16818))


## [1.9.2] - 2023-02-15

- Fixed an attribute error and improved input validation for invalid strategy types being passed to Fabric ([#16693](https://github.com/Lightning-AI/lightning/pull/16693))


## [1.9.1] - 2023-02-10

### Fixed

- Fixed error handling for `accelerator="mps"` and `ddp` strategy pairing ([#16455](https://github.com/Lightning-AI/lightning/pull/16455))
- Fixed strict availability check for `torch_xla` requirement ([#16476](https://github.com/Lightning-AI/lightning/pull/16476))
- Fixed an issue where PL would wrap DataLoaders with XLA's MpDeviceLoader more than once ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed the batch_sampler reference for DataLoaders wrapped with XLA's MpDeviceLoader ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed an import error when `torch.distributed` is not available ([#16658](https://github.com/Lightning-AI/lightning/pull/16658))


## [1.9.0] - 2023-01-17

### Added

- Added `Fabric.launch()` to programmatically launch processes (e.g. in Jupyter notebook) ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))
- Added the option to launch Fabric scripts from the CLI, without the need to wrap the code into the `run` method ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))
- Added `Fabric.setup_module()` and `Fabric.setup_optimizers()` to support strategies that need to set up the model before an optimizer can be created ([#15185](https://github.com/Lightning-AI/lightning/pull/15185))
- Added support for Fully Sharded Data Parallel (FSDP) training in Lightning Lite ([#14967](https://github.com/Lightning-AI/lightning/issues/14967))
- Added `lightning_fabric.accelerators.find_usable_cuda_devices` utility function ([#16147](https://github.com/PyTorchLightning/pytorch-lightning/pull/16147))
- Added basic support for LightningModules ([#16048](https://github.com/Lightning-AI/lightning/issues/16048))
- Added support for managing callbacks via `Fabric(callbacks=...)` and emitting events through `Fabric.call()` ([#16074](https://github.com/Lightning-AI/lightning/issues/16074))
- Added Logger support ([#16121](https://github.com/Lightning-AI/lightning/issues/16121))
  * Added `Fabric(loggers=...)` to support different Logger frameworks in Fabric
  * Added `Fabric.log` for logging scalars using multiple loggers
  * Added `Fabric.log_dict` for logging a dictionary of multiple metrics at once
  * Added `Fabric.loggers` and `Fabric.logger` attributes to access the individual logger instances
  * Added support for calling `self.log` and `self.log_dict` in a LightningModule when using Fabric
  * Added access to `self.logger` and `self.loggers` in a LightningModule when using Fabric
- Added `lightning_fabric.loggers.TensorBoardLogger` ([#16121](https://github.com/Lightning-AI/lightning/issues/16121))
- Added `lightning_fabric.loggers.CSVLogger` ([#16346](https://github.com/Lightning-AI/lightning/issues/16346))
- Added support for a consistent `.zero_grad(set_to_none=...)` on the wrapped optimizer regardless of which strategy is used ([#16275](https://github.com/Lightning-AI/lightning/issues/16275))

### Changed

- Renamed the class `LightningLite` to `Fabric` ([#15932](https://github.com/Lightning-AI/lightning/issues/15932), [#15938](https://github.com/Lightning-AI/lightning/issues/15938))
- The `Fabric.run()` method is no longer abstract ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))
- The `XLAStrategy` now inherits from `ParallelStrategy` instead of `DDPSpawnStrategy` ([#15838](https://github.com/Lightning-AI/lightning/issues/15838))
- Merged the implementation of `DDPSpawnStrategy` into `DDPStrategy` and removed `DDPSpawnStrategy` ([#14952](https://github.com/Lightning-AI/lightning/issues/14952))
- The dataloader wrapper returned from `.setup_dataloaders()` now calls `.set_epoch()` on the distributed sampler if one is used ([#16101](https://github.com/Lightning-AI/lightning/issues/16101))
- Renamed `Strategy.reduce` to `Strategy.all_reduce` in all strategies ([#16370](https://github.com/Lightning-AI/lightning/issues/16370))
- When using multiple devices, the strategy now defaults to "ddp" instead of "ddp_spawn" when none is set ([#16388](https://github.com/Lightning-AI/lightning/issues/16388))

### Removed

- Removed support for FairScale's sharded training (`strategy='ddp_sharded'|'ddp_sharded_spawn'`). Use Fully-Sharded Data Parallel instead (`strategy='fsdp'`) ([#16329](https://github.com/Lightning-AI/lightning/pull/16329))

### Fixed

- Restored sampling parity between PyTorch and Fabric dataloaders when using the `DistributedSampler` ([#16101](https://github.com/Lightning-AI/lightning/issues/16101))
- Fixes an issue where the error message wouldn't tell the user the real value that was passed through the CLI ([#16334](https://github.com/Lightning-AI/lightning/issues/16334))


## [1.8.6] - 2022-12-21

- minor cleaning


## [1.8.5] - 2022-12-15

- minor cleaning


## [1.8.4] - 2022-12-08

### Fixed

- Fixed `shuffle=False` having no effect when using DDP/DistributedSampler ([#15931](https://github.com/Lightning-AI/lightning/issues/15931))


## [1.8.3] - 2022-11-22

### Changed

- Temporarily removed support for Hydra multi-run ([#15737](https://github.com/Lightning-AI/lightning/pull/15737))


## [1.8.2] - 2022-11-17

### Fixed

- Fixed the automatic fallback from `LightningLite(strategy="ddp_spawn", ...)` to `LightningLite(strategy="ddp", ...)` when on an LSF cluster ([#15103](https://github.com/PyTorchLightning/pytorch-lightning/issues/15103))


## [1.8.1] - 2022-11-10

### Fixed

- Fix an issue with the SLURM `srun` detection causing permission errors ([#15485](https://github.com/Lightning-AI/lightning/issues/15485))
- Fixed the import of `lightning_lite` causing a warning 'Redirects are currently not supported in Windows or MacOs' ([#15610](https://github.com/PyTorchLightning/pytorch-lightning/issues/15610))
