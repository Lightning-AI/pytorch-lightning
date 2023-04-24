# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [UnReleased] - 2023-04-DD

### Changed

-


### Fixed

-


## [2.0.2] - 2023-04-24

### Changed

- Enable precision autocast for LightningModule step methods in Fabric ([#17439](https://github.com/Lightning-AI/lightning/pull/17439))

### Fixed

- Fixed an issue with `LightningModule.*_step` methods bypassing the DDP/FSDP wrapper ([#17424](https://github.com/Lightning-AI/lightning/pull/17424))
- Fixed device handling in `Fabric.setup()` when the model has no parameters ([#17441](https://github.com/Lightning-AI/lightning/pull/17441))


## [2.0.1] - 2023-03-30

### Changed

- Generalized `Optimizer` validation to accommodate both FSDP 1.x and 2.x ([#16733](https://github.com/Lightning-AI/lightning/pull/16733))


## [2.0.0] - 2023-03-15

### Added

- Added `Fabric.all_reduce` ([#16459](https://github.com/Lightning-AI/lightning/pull/16459))
- Added support for saving and loading DeepSpeed checkpoints through `Fabric.save/load()` ([#16452](https://github.com/Lightning-AI/lightning/pull/16452))
- Added support for automatically calling `set_epoch` on the `dataloader.batch_sampler.sampler` ([#16841](https://github.com/Lightning-AI/lightning/pull/16841))
- Added support for writing logs to remote file systems with the `CSVLogger` ([#16880](https://github.com/Lightning-AI/lightning/pull/16880))
- Added support for frozen dataclasses in the optimizer state ([#16656](https://github.com/Lightning-AI/lightning/pull/16656))
- Added `lightning.fabric.is_wrapped` to check whether a module, optimizer, or dataloader was already wrapped by Fabric ([#16953](https://github.com/Lightning-AI/lightning/pull/16953))

### Changed

- Fabric now chooses `accelerator="auto", strategy="auto", devices="auto"` as defaults ([#16842](https://github.com/Lightning-AI/lightning/pull/16842))
- Checkpoint saving and loading redesign ([#16434](https://github.com/Lightning-AI/lightning/pull/16434))
  * Changed the method signatrue of `Fabric.save` and `Fabric.load`
  * Changed the method signature of `Strategy.save_checkpoint` and `Fabric.load_checkpoint`
  * `Fabric.save` accepts a state that can contain model and optimizer references
  * `Fabric.load` can now load state in-place onto models and optimizers
  * `Fabric.load` returns a dictionary of objects that weren't loaded into the state
  * `Strategy.save_checkpoint` and `Fabric.load_checkpoint` are now responsible for accessing the state of the model and optimizers
- `DataParallelStrategy.get_module_state_dict()` and `DDPStrategy.get_module_state_dict()` now correctly extracts the state dict without keys prefixed with 'module' ([#16487](https://github.com/Lightning-AI/lightning/pull/16487))
- "Native" suffix removal ([#16490](https://github.com/Lightning-AI/lightning/pull/16490))
  * `strategy="fsdp_full_shard_offload"` is now `strategy="fsdp_cpu_offload"`
  * `lightning.fabric.plugins.precision.native_amp` is now `lightning.fabric.plugins.precision.amp`
- Enabled all shorthand strategy names that can be supported in the CLI ([#16485](https://github.com/Lightning-AI/lightning/pull/16485))
- Renamed `strategy='tpu_spawn'` to `strategy='xla'` and `strategy='tpu_spawn_debug'` to `strategy='xla_debug'` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Changed arguments for precision settings (from [64|32|16|bf16] to ["64-true"|"32-true"|"16-mixed"|"bf16-mixed"]) ([#16767](https://github.com/Lightning-AI/lightning/pull/16767))
- The selection `Fabric(strategy="ddp_spawn", ...)` no longer falls back to "ddp" when a cluster environment gets detected ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- Renamed `setup_dataloaders(replace_sampler=...)` to `setup_dataloaders(use_distributed_sampler=...)` ([#16829](https://github.com/Lightning-AI/lightning/pull/16829))

### Removed

- Removed support for PyTorch 1.10 ([#16492](https://github.com/Lightning-AI/lightning/pull/16492))
- Removed support for Python 3.7 ([#16579](https://github.com/Lightning-AI/lightning/pull/16579))

### Fixed

- Fixed issue where the wrapped dataloader `iter()` would be called twice ([#16841](https://github.com/Lightning-AI/lightning/pull/16841))

- Improved the error message for installing tensorboard or tensorboardx ([#17053](https://github.com/Lightning-AI/lightning/pull/17053))


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

### Fixed

- Fixed an attribute error and improved input validation for invalid strategy types being passed to Trainer ([#16693](https://github.com/Lightning-AI/lightning/pull/16693))


## [1.9.1] - 2023-02-10

### Fixed

- Fixed error handling for `accelerator="mps"` and `ddp` strategy pairing ([#16455](https://github.com/Lightning-AI/lightning/pull/16455))
- Fixed strict availability check for `torch_xla` requirement ([#16476](https://github.com/Lightning-AI/lightning/pull/16476))
- Fixed an issue where PL would wrap DataLoaders with XLA's MpDeviceLoader more than once ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed the batch_sampler reference for DataLoaders wrapped with XLA's MpDeviceLoader ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed an import error when `torch.distributed` is not available ([#16658](https://github.com/Lightning-AI/lightning/pull/16658))


## [1.9.0] - 2023-01-17

### Added

- Added `Fabric.launch()` to programmatically launch processes (e.g. in Jupyter notebook) ([#14992](https://github.com/Lightning-AI/lightning/pull/14992))
- Added the option to launch Fabric scripts from the CLI, without the need to wrap the code into the `run` method ([#14992](https://github.com/Lightning-AI/lightning/pull/14992))
- Added `Fabric.setup_module()` and `Fabric.setup_optimizers()` to support strategies that need to set up the model before an optimizer can be created ([#15185](https://github.com/Lightning-AI/lightning/pull/15185))
- Added support for Fully Sharded Data Parallel (FSDP) training in Lightning Lite ([#14967](https://github.com/Lightning-AI/lightning/pull/14967))
- Added `lightning.fabric.accelerators.find_usable_cuda_devices` utility function ([#16147](https://github.com/Lightning-AI/lightning/pull/16147))
- Added basic support for LightningModules ([#16048](https://github.com/Lightning-AI/lightning/pull/16048))
- Added support for managing callbacks via `Fabric(callbacks=...)` and emitting events through `Fabric.call()` ([#16074](https://github.com/Lightning-AI/lightning/pull/16074))
- Added Logger support ([#16121](https://github.com/Lightning-AI/lightning/pull/16121))
  * Added `Fabric(loggers=...)` to support different Logger frameworks in Fabric
  * Added `Fabric.log` for logging scalars using multiple loggers
  * Added `Fabric.log_dict` for logging a dictionary of multiple metrics at once
  * Added `Fabric.loggers` and `Fabric.logger` attributes to access the individual logger instances
  * Added support for calling `self.log` and `self.log_dict` in a LightningModule when using Fabric
  * Added access to `self.logger` and `self.loggers` in a LightningModule when using Fabric
- Added `lightning.fabric.loggers.TensorBoardLogger` ([#16121](https://github.com/Lightning-AI/lightning/pull/16121))
- Added `lightning.fabric.loggers.CSVLogger` ([#16346](https://github.com/Lightning-AI/lightning/pull/16346))
- Added support for a consistent `.zero_grad(set_to_none=...)` on the wrapped optimizer regardless of which strategy is used ([#16275](https://github.com/Lightning-AI/lightning/pull/16275))

### Changed

- Renamed the class `LightningLite` to `Fabric` ([#15932](https://github.com/Lightning-AI/lightning/pull/15932), [#15938](https://github.com/Lightning-AI/lightning/pull/15938))
- The `Fabric.run()` method is no longer abstract ([#14992](https://github.com/Lightning-AI/lightning/pull/14992))
- The `XLAStrategy` now inherits from `ParallelStrategy` instead of `DDPSpawnStrategy` ([#15838](https://github.com/Lightning-AI/lightning/pull/15838))
- Merged the implementation of `DDPSpawnStrategy` into `DDPStrategy` and removed `DDPSpawnStrategy` ([#14952](https://github.com/Lightning-AI/lightning/pull/14952))
- The dataloader wrapper returned from `.setup_dataloaders()` now calls `.set_epoch()` on the distributed sampler if one is used ([#16101](https://github.com/Lightning-AI/lightning/pull/16101))
- Renamed `Strategy.reduce` to `Strategy.all_reduce` in all strategies ([#16370](https://github.com/Lightning-AI/lightning/pull/16370))
- When using multiple devices, the strategy now defaults to "ddp" instead of "ddp_spawn" when none is set ([#16388](https://github.com/Lightning-AI/lightning/pull/16388))

### Removed

- Removed support for FairScale's sharded training (`strategy='ddp_sharded'|'ddp_sharded_spawn'`). Use Fully-Sharded Data Parallel instead (`strategy='fsdp'`) ([#16329](https://github.com/Lightning-AI/lightning/pull/16329))

### Fixed

- Restored sampling parity between PyTorch and Fabric dataloaders when using the `DistributedSampler` ([#16101](https://github.com/Lightning-AI/lightning/pull/16101))
- Fixes an issue where the error message wouldn't tell the user the real value that was passed through the CLI ([#16334](https://github.com/Lightning-AI/lightning/pull/16334))


## [1.8.6] - 2022-12-21

- minor cleaning


## [1.8.5] - 2022-12-15

- minor cleaning


## [1.8.4] - 2022-12-08

### Fixed

- Fixed `shuffle=False` having no effect when using DDP/DistributedSampler ([#15931](https://github.com/Lightning-AI/lightning/pull/15931))


## [1.8.3] - 2022-11-22

### Changed

- Temporarily removed support for Hydra multi-run ([#15737](https://github.com/Lightning-AI/lightning/pull/15737))


## [1.8.2] - 2022-11-17

### Fixed

- Fixed the automatic fallback from `LightningLite(strategy="ddp_spawn", ...)` to `LightningLite(strategy="ddp", ...)` when on an LSF cluster ([#15103](https://github.com/Lightning-AI/lightning/pull/15103))


## [1.8.1] - 2022-11-10

### Fixed

- Fix an issue with the SLURM `srun` detection causing permission errors ([#15485](https://github.com/Lightning-AI/lightning/pull/15485))
- Fixed the import of `lightning_lite` causing a warning 'Redirects are currently not supported in Windows or MacOs' ([#15610](https://github.com/Lightning-AI/lightning/pull/15610))
