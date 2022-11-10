# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [unreleased] - 202Y-MM-DD


### Added


- Added `LightningLite.launch()` to programmatically launch processes (e.g. in Jupyter notebook) ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))
- Added the option to launch Lightning Lite scripts from the CLI, without the need to wrap the code into the `run` method ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))

-

-


### Changed

- The `LightningLite.run()` method is no longer abstract ([#14992](https://github.com/Lightning-AI/lightning/issues/14992))

-

-


### Deprecated

-

-

-


### Removed

-

-

-


### Fixed

- Fix an issue with the SLURM `srun` detection causing permission errors ([#15485](https://github.com/Lightning-AI/lightning/issues/15485))

- Fixed the import of `lightning_lite` causing a warning 'Redirects are currently not supported in Windows or MacOs' ([#15610](https://github.com/PyTorchLightning/pytorch-lightning/issues/15610))

-
