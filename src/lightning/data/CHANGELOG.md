# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [2.1.3] - 2023-12-21

### Added

- Add fault tolerance `StreamingDataset` ([#19052](https://github.com/Lightning-AI/lightning/pull/19052))
- Add numpy support for the `StreamingDataset` ([#19050](https://github.com/Lightning-AI/lightning/pull/19050))
- Add fault tolerance for the `StreamingDataset` ([#19049](https://github.com/Lightning-AI/lightning/pull/19049))
- Add direct s3 support to the `StreamingDataset` ([#19044](https://github.com/Lightning-AI/lightning/pull/19044))
- Add disk usage check before downloading files ([#19041](https://github.com/Lightning-AI/lightning/pull/19041))

### Changed

- Cleanup chunks right away if the dataset doesn't fit within the cache in `StreamingDataset` ([#19168](https://github.com/Lightning-AI/lightning/pull/19168))
- `StreamingDataset` improve deletion strategy ([#19118](https://github.com/Lightning-AI/lightning/pull/19118))
- Improve `StreamingDataset` Speed ([#19114](https://github.com/Lightning-AI/lightning/pull/19114))
- Remove time in the Data Processor progress bar ([#19108](https://github.com/Lightning-AI/lightning/pull/19108))
- Optimize loading time for chunks to be there ([#19109](https://github.com/Lightning-AI/lightning/pull/19109))
- Resolve path for `StreamingDataset` ([#19094](https://github.com/Lightning-AI/lightning/pull/19094))
- Make input dir in `DataProcessor` required ([#18910](https://github.com/Lightning-AI/lightning/pull/18910))
- Remove the `LightningDataset` relying on un-maintained torchdata ([#19019](https://github.com/Lightning-AI/lightning/pull/19019))

# Fixed

- Resolve checkpointing for the Streaming Dataset ([#19123](https://github.com/Lightning-AI/lightning/pull/19123))
- Resolve Item Loader bugs ([#19017](https://github.com/Lightning-AI/lightning/pull/19017))


## [2.1.2] - 2023-11-15

### Added

- Created cache dir if it doesn't exist ([#18955](https://github.com/Lightning-AI/lightning/pull/18955))
- Cached directory per worker to avoid collisions ([#18957](https://github.com/Lightning-AI/lightning/pull/18957))
- Added the input_dir in the cache_dir to avoid overlapping downloads ([#18960](https://github.com/Lightning-AI/lightning/pull/18960))
- Added support for deleting chunks ([#18959](https://github.com/Lightning-AI/lightning/pull/18959))
- Added Video/Audio support ([#18977](https://github.com/Lightning-AI/lightning/pull/18977))
- Added multiple uploaders to the map, optimize ([#18989](https://github.com/Lightning-AI/lightning/pull/18989))

### Changed

- Greedily select files for data processor workers based on size ([#18907](https://github.com/Lightning-AI/lightning/pull/18907))
- Prevented downloading more chunks than needed ([#18964](https://github.com/Lightning-AI/lightning/pull/18964))


## [2.1.1] - 2023-11-06

### Added

- Added name and version ([#18796](https://github.com/Lightning-AI/lightning/pull/18796))
- Added support for text ([#18807](https://github.com/Lightning-AI/lightning/pull/18807))
- Introduced Dataset Optimizer (
    [#18788](https://github.com/Lightning-AI/lightning/pull/18788),
    [#18817](https://github.com/Lightning-AI/lightning/pull/18817),
    [#18827](https://github.com/Lightning-AI/lightning/pull/18827)
)
- Added distributed support for StreamingDataset ([#18850](https://github.com/Lightning-AI/lightning/pull/18850))
- Added broadcast to Dataset Optimizer with multiple nodes ([#18860](https://github.com/Lightning-AI/lightning/pull/18860))
- Improved Streaming Dataset API ([#18882](https://github.com/Lightning-AI/lightning/pull/18882))
- Prevent leaking the thread to the workers ([#18891](https://github.com/Lightning-AI/lightning/pull/18891))
- Added DataRecipe ([#18892](https://github.com/Lightning-AI/lightning/pull/18892))
- Improved map and chunkify ([#18901](https://github.com/Lightning-AI/lightning/pull/18901))
- Added human-readable format for chunk_bytes ([#18925](https://github.com/Lightning-AI/lightning/pull/18925))
- Improved s3 client support ([#18920](https://github.com/Lightning-AI/lightning/pull/18920))
- Added dataset creation ([#18940](https://github.com/Lightning-AI/lightning/pull/18940))


## [2.1.0] - 2023-10-11

### Added

- Added `LightningDataset` for optimized data loading including fast loading for S3 buckets. ([#17743](https://github.com/Lightning-AI/lightning/pull/17743))
- Added `LightningIterableDataset` for resumable dataloading with iterable datasets ([#17998](https://github.com/Lightning-AI/lightning/pull/17998))
