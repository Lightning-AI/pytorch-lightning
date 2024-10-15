# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.11.8] - 2024-10-15

### Changed

- CI: enable custom env. vars in pkg check workflow ([#317](https://github.com/Lightning-AI/utilities/pull/317))


## [0.11.7] - 2024-09-02

### Fixed

- CI: pass `include-hidden-files: true` to upload created packages ([#303](https://github.com/Lightning-AI/utilities/pull/303))


## [0.11.6] - 2024-07-23

### Changed

- CI: rename `import-extras` to `custom-import` in package check ([#287](https://github.com/Lightning-AI/utilities/pull/287))

### Fixed

- CI: update type/`mypy` check ([#288](https://github.com/Lightning-AI/utilities/pull/288))
- Fixed parsing pre-release package versions in `RequirementCache` ([#292](https://github.com/Lightning-AI/utilities/pull/292))


## [0.11.5] - 2024-07-15

### Fixed

- Fixed extras check in RequirementCache ([#283](https://github.com/Lightning-AI/utilities/pull/283))


## [0.11.4] - 2024-07-14

### Changed

- Replaced deprecated `pkg_resources` with `importlib.metadata` ([#281](https://github.com/Lightning-AI/utilities/pull/281))


## [0.11.3] - 2024-06-26

### Fixed

- CI: freeze tools for Pkg action ([#273](https://github.com/Lightning-AI/utilities/pull/273))


## [0.11.2] - 2024-03-28

### Fixed

- docs: fix parsing non-trivial package name ([#247](https://github.com/Lightning-AI/utilities/pull/247))


## [0.11.1] - 2024-03-25

### Changed

- CI: enable setting python version for package build ([#244](https://github.com/Lightning-AI/utilities/pull/244))
- docs: fix/use PyPI versions for pinning links ([#243](https://github.com/Lightning-AI/utilities/pull/243))

### Fixed

- docs: do not replace external link for self ([#245](https://github.com/Lightning-AI/utilities/pull/245))


## [0.11.0] - 2024-03-18

### Added

- docs: enable pin version in links to external docs ([#236](https://github.com/Lightning-AI/utilities/pull/236))

### Changed

- CI: parametrize source folder for typing check ([#228](https://github.com/Lightning-AI/utilities/pull/228))

---

## [0.10.1] - 2023-12-22

### Fixed

- Avoid accidental namedtuple conversion in `apply_to_collection` ([#210](https://github.com/Lightning-AI/utilities/pull/210))


## [0.10.0] - 2023-11-17

### Added

- CI: added `install-extras` in install check allowing deduplication eventual circular install dependency (
    [#184](https://github.com/Lightning-AI/utilities/pull/184),
    [#185](https://github.com/Lightning-AI/utilities/pull/185)
)
- Added `rank_zero_only(..., default=?)` argument to return a default value on rank > 1 ([#187](https://github.com/Lightning-AI/utilities/pull/187))

### Changed

- Updated/Extended the `requires` wrapper ([#146](https://github.com/Lightning-AI/utilities/pull/146))
- CI: updated/extended cleaning old and/or specific caches ([#159](https://github.com/Lightning-AI/utilities/pull/159))
- CI: unified/extended docs makes flows ([#162](https://github.com/Lightning-AI/utilities/pull/162))
- CI: allow Other targets for building docs ([#179](https://github.com/Lightning-AI/utilities/pull/179))
- CI: narrow scope for md links check ([#183](https://github.com/Lightning-AI/utilities/pull/183))
- CI: split code checks & enable pre-commit updates (
    [#191](https://github.com/Lightning-AI/utilities/pull/191),
    [#193](https://github.com/Lightning-AI/utilities/pull/193),
    [#194](https://github.com/Lightning-AI/utilities/pull/194)
)

### Deprecated

- Deprecated `ModuleAvailableCache` in favor of `RequirementCache` ([#147](https://github.com/Lightning-AI/utilities/pull/147))

### Fixed

- Fixed issue with `is_overridden` falsely returning True when the parent method is wrapped ([#149](https://github.com/Lightning-AI/utilities/pull/149))
- CI: optional freeze version of schema check ([#148](https://github.com/Lightning-AI/utilities/pull/148))
- CI: fixed guard for `pkg-check` workflow on canceled  ([#180](https://github.com/Lightning-AI/utilities/pull/180))
- CI: resolve latex dependency for docs builds ([#181](https://github.com/Lightning-AI/utilities/pull/181))
- CI: fixed branch for md links check ([#183](https://github.com/Lightning-AI/utilities/pull/183))

---

## [0.9.0] - 2023-06-29

### Added

- docs: fetch all external resources for building docs ([#142](https://github.com/Lightning-AI/utilities/pull/142))

### Changed

- CI: allow splitting docs's tests and make ([#141](https://github.com/Lightning-AI/utilities/pull/141))

### Fixed

- Fixed - do not erase function types in decorators ([#135](https://github.com/Lightning-AI/utilities/pull/135))
- CI: fix passing install flags in package check ([#137](https://github.com/Lightning-AI/utilities/pull/137))

---

## [0.8.0] - 2023-03-10

### Added

- Added requirements parser ([#107](https://github.com/Lightning-AI/utilities/pull/107))
- Added workflow for checking markdown links ([#81](https://github.com/Lightning-AI/utilities/pull/81))

---

## [0.7.1] - 2023-02-23

### Added

- CI: guardian as parametrization closure ([#111](https://github.com/Lightning-AI/utilities/pull/111))

### Changed

- CI: allow to specify typing extra ([#110](https://github.com/Lightning-AI/utilities/pull/110))

### Fixed

- More resilient `RequirementCache` that checks for module import-ability ([#112](https://github.com/Lightning-AI/utilities/pull/112))


## [0.7.0] - 2023-02-20

### Added

- Allow frozen dataclasses in `apply_to_collection` ([#98](https://github.com/Lightning-AI/utilities/pull/98))
- Extended `StrEnum.from_str` with optional raising ValueError ([#99](https://github.com/Lightning-AI/utilities/pull/99))


### Changed

- CI/docs: allow passing env. variables ([#96](https://github.com/Lightning-AI/utilities/pull/96))
- CI: build package ([#104](https://github.com/Lightning-AI/utilities/pull/104))


### Fixed

- Fixed `StrEnum.from_str` with source as key (
    [#99](https://github.com/Lightning-AI/utilities/pull/99),
    [#102](https://github.com/Lightning-AI/utilities/pull/102)
)

---

## [0.6.0] - 2023-01-23

### Added

- Added `ModuleAvailableCache` ([#86](https://github.com/Lightning-AI/utilities/pull/86))

### Changed

- Apply local actions in reusable workflows ([#51](https://github.com/Lightning-AI/utilities/pull/51))
- CI: abstract package actions ([#48](https://github.com/Lightning-AI/utilities/pull/48))
- CI: Checkout submodules recursive ([#82](https://github.com/Lightning-AI/utilities/pull/82))

### Fixed

- CI: Checking scheme in both yaml & yml + verbose ([#84](https://github.com/Lightning-AI/utilities/pull/84))

---

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

---

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

---

## [0.3.0] - 2022-09-06

### Added

- Added `StrEnum` class ([#38](https://github.com/Lightning-AI/utilities/pull/38))
- Added rank-zero utilities ([#36](https://github.com/Lightning-AI/utilities/pull/36))
- Added `is_overridden` utilities ([#35](https://github.com/Lightning-AI/utilities/pull/35))
- Added `get_all_subclasses` ([#39](https://github.com/Lightning-AI/utilities/pull/39))

---

## [0.2.0] - 2022-09-05

### Added

- Added core and dev directories ([#28](https://github.com/Lightning-AI/utilities/pull/28))
- Added import utilities ([#20](https://github.com/Lightning-AI/utilities/pull/20))
- Added import caches ([#21](https://github.com/Lightning-AI/utilities/pull/21))
- Added `apply_func` utilities ([#32](https://github.com/Lightning-AI/utilities/pull/32))

### Changed

- Renamed `pl-devtools` -> `lightning_utilities` ([#27](https://github.com/Lightning-AI/utilities/pull/27), [#30](https://github.com/Lightning-AI/utilities/pull/30))

---

## [0.1.0] - 2022-08-22

### Added

- Added initial reusable workflows and actions and fix any failures ([#2](https://github.com/Lightning-AI/utilities/pull/2))
- Added actions: cache + tests ([#5](https://github.com/Lightning-AI/utilities/pull/5))
- Added basic package functionality with CLI ([#3](https://github.com/Lightning-AI/utilities/pull/3))
