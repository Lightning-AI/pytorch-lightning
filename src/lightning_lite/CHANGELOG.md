# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [unreleased] - 202Y-MM-DD


### Added


- Added `Fabric.launch()` to programmatically launch processes (e.g. in Jupyter notebook) ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))


- Added the option to launch Lightning Lite scripts from the CLI, without the need to wrap the code into the `run` method ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))


- Added `Fabric.setup_module()` and `Fabric.setup_optimizers()` to support strategies that need to set up the model before an optimizer can be created ([#15185](https://github.com/Lightning-AI/lightning/pull/15185))


- Added support for Fully Sharded Data Parallel (FSDP) training in Lightning Lite ([#14967](https://github.com/Lightning-AI/lightning/issues/14967))


### Changed

- The `Fabric.run()` method is no longer abstract ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))


-


### Deprecated

-


### Removed

-

### Fixed

-


## [1.8.3] - 2022-11-22

### Changed

- Temporarily removed support for Hydra multi-run ([#15737](https://github.com/Lightning-AI/lightning/pull/15737))


## [1.8.2] - 2022-11-17

### Fixed

- Fixed the automatic fallback from `Fabric(strategy="ddp_spawn", ...)` to `Fabric(strategy="ddp", ...)` when on an LSF cluster ([#15103](https://github.com/PyTorchLightning/pytorch-lightning/issues/15103))


## [1.8.1] - 2022-11-10

### Fixed

- Fix an issue with the SLURM `srun` detection causing permission errors ([#15485](https://github.com/Lightning-AI/lightning/issues/15485))
- Fixed the import of `lightning_lite` causing a warning 'Redirects are currently not supported in Windows or MacOs' ([#15610](https://github.com/PyTorchLightning/pytorch-lightning/issues/15610))
