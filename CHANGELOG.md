# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2022-12-21

### Added

- Added method to lazily import modules ([#71](https://github.com/Lightning-AI/utilities/pull/71))
- Added `requires` wrapper ([#70](https://github.com/Lightning-AI/utilities/pull/70))
- Added several functions/class common to package's `__all__` ([#76](https://github.com/Lightning-AI/utilities/pull/76))


### Changed

- CI: extended package install check ([#76](https://github.com/Lightning-AI/utilities/pull/76))
- Allowed `StrEnum.from_str` by values ([#77](https://github.com/Lightning-AI/utilities/pull/77))

### Fixed

- Fixed requirements parsing ([#69](https://github.com/Lightning-AI/utilities/pull/69))
- Fixed missing `packaging` dependency ([#76](https://github.com/Lightning-AI/utilities/pull/76))


## [0.4.2] - 2022-10-31

### Fixed

- Fixed MANIFEST ([#68](https://github.com/Lightning-AI/utilities/pull/68))


## [0.4.1] - 2022-10-31

### Fixed

- Fixed cannot import name `metadata` from `importlib` ([#65](https://github.com/Lightning-AI/utilities/pull/65))

## [0.4.0] - 2022-10-27

### Added

- Added pip list action ([#17](https://github.com/Lightning-AI/utilities/pull/17))
- Added reusable workflow to clear caches within a repository ([#43](https://github.com/Lightning-AI/utilities/pull/43))
- Added require input for docs workflow ([#50](https://github.com/Lightning-AI/utilities/pull/50))
- Added `lightning_utilities.test.warning.no_warning_call` ([#55](https://github.com/Lightning-AI/utilities/pull/55))

### Changed

- Moved CLI dependencies to package's extra installation ([#42](https://github.com/Lightning-AI/utilities/pull/42))
  `fire` added to `lightning_tools[dev]`
- Increased verbosity and comment schema file location ([#49](https://github.com/Lightning-AI/utilities/pull/49))
- Renamed `lightning_utilities.dev` to `lightning_utilities.cli` ([#46](https://github.com/Lightning-AI/utilities/pull/46))


## [0.3.0] - 2022-09-06

### Added

- Added `StrEnum` class ([#38](https://github.com/Lightning-AI/utilities/pull/38))
- Added rank-zero utilities ([#36](https://github.com/Lightning-AI/utilities/pull/36))
- Added `is_overridden` utilities ([#35](https://github.com/Lightning-AI/utilities/pull/35))
- Added `get_all_subclasses` ([#39](https://github.com/Lightning-AI/utilities/pull/39))


## [0.2.0] - 2022-09-05

### Added

- Added core and dev directories ([#28](https://github.com/Lightning-AI/utilities/pull/28))
- Added import utilities ([#20](https://github.com/Lightning-AI/utilities/pull/20))
- Added import caches ([#21](https://github.com/Lightning-AI/utilities/pull/21))
- Added `apply_func` utilities ([#32](https://github.com/Lightning-AI/utilities/pull/32))

### Changed

- Renamed `pl-devtools` -> `lightning_utilities` ([#27](https://github.com/Lightning-AI/utilities/pull/27), [#30](https://github.com/Lightning-AI/utilities/pull/30))


## [0.1.0] - 2022-08-22

### Added

- Added initial reusable workflows and actions and fix any failures ([#2](https://github.com/Lightning-AI/utilities/pull/2))
- Added actions: cache + tests ([#5](https://github.com/Lightning-AI/utilities/pull/5))
- Added basic package functionality with CLI ([#3](https://github.com/Lightning-AI/utilities/pull/3))
