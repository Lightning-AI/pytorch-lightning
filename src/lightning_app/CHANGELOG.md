# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.6.0] - 2022-MM-DD

### Added

- Add support for `Lightning App Commands` through the `configure_commands` hook on the Lightning Flow and the `ClientCommand`  ([#13602](https://github.com/Lightning-AI/lightning/pull/13602))

- Adds `LightningTrainingComponent`. `LightningTrainingComponent` orchestrates multi-node training in the cloud ([#13830](https://github.com/Lightning-AI/lightning/pull/13830))

- Add support for `Lightning API` through the `configure_api` hook on the Lightning Flow and the `Post`, `Get`, `Delete`, `Put` HttpMethods ([#13945](https://github.com/Lightning-AI/lightning/pull/13945))
### Changed

- Update the Lightning App docs ([#13537](https://github.com/Lightning-AI/lightning/pull/13537))

### Changed

- Added `LIGHTNING_` prefix to Platform AWS credentials ([#13703](https://github.com/Lightning-AI/lightning/pull/13703))

### Deprecated

### Fixed
