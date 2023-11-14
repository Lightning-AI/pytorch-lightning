# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [UnReleased] - 2023-11-DD

### Added

-


### Fixed

-


## [2.1.1] - 2023-11-06

### Added

-  Add name and version ([#18796](https://github.com/Lightning-AI/lightning/pull/18796))
- Add support for text ([#18807](https://github.com/Lightning-AI/lightning/pull/18807))
- Introduce Dataset Optimizer (
[#18788](https://github.com/Lightning-AI/lightning/pull/18788),
[#18817](https://github.com/Lightning-AI/lightning/pull/18817),
[#18827](https://github.com/Lightning-AI/lightning/pull/18827))
- Add distributed support for StreamingDataset ([#18850](https://github.com/Lightning-AI/lightning/pull/18850))
- Add broadcast to Dataset Optimizer with multiple nodes ([#18860](https://github.com/Lightning-AI/lightning/pull/18860))
- Improve Streaming Dataset API ([#18882](https://github.com/Lightning-AI/lightning/pull/18882))
- Prevent leaking the thread to the workers ([#18891](https://github.com/Lightning-AI/lightning/pull/18891))
- Add DataRecipe ([#18892](https://github.com/Lightning-AI/lightning/pull/18892))
- Improve map and chunkify ([#18901](https://github.com/Lightning-AI/lightning/pull/18901))
- Add human readable format for chunk_bytes ([#18925](https://github.com/Lightning-AI/lightning/pull/18925))
- Improve s3 client support ([#18920](https://github.com/Lightning-AI/lightning/pull/18920))
- Add dataset creation ([#18940](https://github.com/Lightning-AI/lightning/pull/18940))


## [2.1.0] - 2023-10-11

### Added

- Added `LightningDataset` for optimized data loading including fast loading for S3 buckets. ([#17743](https://github.com/Lightning-AI/lightning/pull/17743))
- Added `LightningIterableDataset` for resumable dataloading with iterable datasets ([#17998](https://github.com/Lightning-AI/lightning/pull/17998))
