# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [UnReleased] - 2022-11-DD

### Added

- Added title and description to ServeGradio ([#15639](https://github.com/Lightning-AI/lightning/pull/15639))
- Added a friendly error message when attempting to run the default cloud compute with a custom base image configured ([#14929](https://github.com/Lightning-AI/lightning/pull/14929))

### Changed

- Improved support for running apps when dependencies aren't installed ([#15711](https://github.com/Lightning-AI/lightning/pull/15711))
- Changed the root directory of the app (which gets uploaded) to be the folder containing the app file, rather than any parent folder containing a `.lightning` file ([#15654](https://github.com/Lightning-AI/lightning/pull/15654))
- Enabled MultiNode Components to support state broadcasting ([#15607](https://github.com/Lightning-AI/lightning/pull/15607))
- Prevent artefactual "running from outside your current environment" error ([#15647](https://github.com/Lightning-AI/lightning/pull/15647))
- Rename failed -> error in tables ([#15608](https://github.com/Lightning-AI/lightning/pull/15608))

### Fixed

- Fixed race condition to over-write the frontend with app infos ([#15398](https://github.com/Lightning-AI/lightning/pull/15398))
- Fixed bi-directional queues sending delta with Drive Component name changes ([#15642](https://github.com/Lightning-AI/lightning/pull/15642))
- Fixed CloudRuntime works collection with structures and accelerated multi node startup time ([#15650](https://github.com/Lightning-AI/lightning/pull/15650))
- Fixed catimage import ([#15712](https://github.com/Lightning-AI/lightning/pull/15712))
- Parse all lines in app file looking for shebangs to run commands ([#15714](https://github.com/Lightning-AI/lightning/pull/15714))


## [1.8.1] - 2022-11-10

### Added

- Added the `start` method to the work ([#15523](https://github.com/Lightning-AI/lightning/pull/15523))
- Added a `MultiNode` Component to run with distributed computation with any frameworks ([#15524](https://github.com/Lightning-AI/lightning/pull/15524))
- Expose `RunWorkExecutor` to the work and provides default ones for the `MultiNode` Component ([#15561](https://github.com/Lightning-AI/lightning/pull/15561))
- Added a `start_with_flow` flag to the `LightningWork` which can be disabled to prevent the work from starting at the same time as the flow ([#15591](https://github.com/Lightning-AI/lightning/pull/15591))
- Added support for running Lightning App with VSCode IDE debugger ([#15590](https://github.com/Lightning-AI/lightning/pull/15590))
- Added `bi-directional` delta updates between the flow and the works ([#15582](https://github.com/Lightning-AI/lightning/pull/15582))
- Added `--setup` flag to `lightning run app` CLI command allowing for dependency installation via app comments ([#15577](https://github.com/Lightning-AI/lightning/pull/15577))
- Auto-upgrade / detect environment mis-match from the CLI ([#15434](https://github.com/Lightning-AI/lightning/pull/15434))
- Added Serve component ([#15609](https://github.com/Lightning-AI/lightning/pull/15609))


### Changed

- Changed the `flow.flows` to be recursive wont to align the behavior with the `flow.works` ([#15466](https://github.com/Lightning-AI/lightning/pull/15466))
- The `params` argument in `TracerPythonScript.run` no longer prepends `--` automatically to parameters ([#15518](https://github.com/Lightning-AI/lightning/pull/15518))
- Only check versions / env when not in the cloud ([#15504](https://github.com/Lightning-AI/lightning/pull/15504))
- Periodically sync database to the drive ([#15441](https://github.com/Lightning-AI/lightning/pull/15441))
- Slightly safer multi node ([#15538](https://github.com/Lightning-AI/lightning/pull/15538))
- Reuse existing commands when running connect more than once ([#15471](https://github.com/Lightning-AI/lightning/pull/15471))

### Fixed

- Fixed writing app name and id in connect.txt file for the command CLI ([#15443](https://github.com/Lightning-AI/lightning/pull/15443))
- Fixed missing root flow among the flows of the app ([#15531](https://github.com/Lightning-AI/lightning/pull/15531))
- Fixed bug with Multi Node Component and add some examples ([#15557](https://github.com/Lightning-AI/lightning/pull/15557))
- Fixed a bug where payload would take a very long time locally ([#15557](https://github.com/Lightning-AI/lightning/pull/15557))
- Fixed an issue with the `lightning` CLI taking a long time to error out when the cloud is not reachable ([#15412](https://github.com/Lightning-AI/lightning/pull/15412))


## [1.8.0] - 2022-11-01

### Added

- Added `load_state_dict` and `state_dict` hooks for `LightningFlow` components ([#14100](https://github.com/Lightning-AI/lightning/pull/14100))
- Added a `--secret` option to CLI to allow binding secrets to app environment variables when running in the cloud ([#14612](https://github.com/Lightning-AI/lightning/pull/14612))
- Added support for running the works without cloud compute in the default container ([#14819](https://github.com/Lightning-AI/lightning/pull/14819))
- Added an HTTPQueue as an optional replacement for the default redis queue ([#14978](https://github.com/Lightning-AI/lightning/pull/14978)
- Added support for configuring flow cloud compute ([#14831](https://github.com/Lightning-AI/lightning/pull/14831))
- Added support for adding descriptions to commands either through a docstring or the `DESCRIPTION` attribute ([#15193](https://github.com/Lightning-AI/lightning/pull/15193)
- Added a try / catch mechanism around request processing to avoid killing the flow ([#15187](https://github.com/Lightning-AI/lightning/pull/15187)
- Added an Database Component ([#14995](https://github.com/Lightning-AI/lightning/pull/14995)
- Added authentication to HTTP queue ([#15202](https://github.com/Lightning-AI/lightning/pull/15202))
- Added support to pass a `LightningWork` to the `LightningApp` ([#15215](https://github.com/Lightning-AI/lightning/pull/15215)
- Added support getting CLI help for connected apps even if the app isn't running ([#15196](https://github.com/Lightning-AI/lightning/pull/15196)
- Added support for adding requirements to commands and installing them when missing when running an app command ([#15198](https://github.com/Lightning-AI/lightning/pull/15198)
- Added Lightning CLI Connection to be terminal session instead of global ([#15241](https://github.com/Lightning-AI/lightning/pull/15241)
- Added support for managing SSH-keys via CLI ([#15291](https://github.com/Lightning-AI/lightning/pull/15291))
- Add a `JustPyFrontend` to ease UI creation with `https://github.com/justpy-org/justpy` ([#15002](https://github.com/Lightning-AI/lightning/pull/15002))
- Added a layout endpoint to the Rest API and enable to disable pulling or pushing to the state ([#15367](https://github.com/Lightning-AI/lightning/pull/15367)
- Added support for functions for `configure_api` and `configure_commands` to be executed in the Rest API process ([#15098](https://github.com/Lightning-AI/lightning/pull/15098)
- Added support for accessing Lighting Apps via SSH ([#15310](https://github.com/Lightning-AI/lightning/pull/15310))
- Added support to start lightning app on cloud without needing to install dependencies locally ([#15019](https://github.com/Lightning-AI/lightning/pull/15019)

### Changed

- Improved the show logs command to be standalone and re-usable ([#15343](https://github.com/Lightning-AI/lightning/pull/15343)
- Removed the `--instance-types` option when creating clusters ([#15314](https://github.com/Lightning-AI/lightning/pull/15314))

### Fixed

- Fixed an issue when using the CLI without arguments ([#14877](https://github.com/Lightning-AI/lightning/pull/14877))
- Fixed a bug where the upload files endpoint would raise an error when running locally ([#14924](https://github.com/Lightning-AI/lightning/pull/14924))
- Fixed BYOC cluster region selector -> hiding it from help since only us-east-1 has been tested and is recommended ([#15277]https://github.com/Lightning-AI/lightning/pull/15277)
- Fixed a bug when launching an app on multiple clusters ([#15226](https://github.com/Lightning-AI/lightning/pull/15226))
- Fixed a bug with a default CloudCompute for Lightning flows ([#15371](https://github.com/Lightning-AI/lightning/pull/15371))

## [0.6.2] - 2022-09-21

### Changed

- Improved Lightning App connect logic by disconnecting automatically ([#14532](https://github.com/Lightning-AI/lightning/pull/14532))
- Improved the error message when the `LightningWork` is missing the `run` method ([#14759](https://github.com/Lightning-AI/lightning/pull/14759))
- Improved the error message when the root `LightningFlow` passed to `LightningApp` is missing the `run` method ([#14760](https://github.com/Lightning-AI/lightning/pull/14760))

### Fixed

- Fixed a bug where the uploaded command file wasn't properly parsed ([#14532](https://github.com/Lightning-AI/lightning/pull/14532))
- Fixed an issue where custom property setters were not being used `LightningWork` class ([#14259](https://github.com/Lightning-AI/lightning/pull/14259))
- Fixed an issue where some terminals would display broken icons in the PL app CLI ([#14226](https://github.com/Lightning-AI/lightning/pull/14226))


## [0.6.1] - 2022-09-19

### Added

- Add support to upload files to the Drive through an asynchronous `upload_file` endpoint ([#14703](https://github.com/Lightning-AI/lightning/pull/14703))

### Changed

- Application storage prefix moved from `app_id` to `project_id/app_id` (#14583)
- LightningCloud client calls to use keyword arguments instead of positional arguments (#14685)

### Fixed

- Making `threadpool` non-default from LightningCloud client  (#14757)
- Resolved a bug where the state change detection using DeepDiff won't work with Path, Drive objects (#14465)
- Resolved a bug where the wrong client was passed to collect cloud logs (#14684)
- Resolved the memory leak issue with the Lightning Cloud package and bumped the requirements to use the latest version (#14697)
- Fixing 5000 log line limitation for Lightning AI BYOC cluster logs (#14458)
- Fixed a bug where the uploaded command file wasn't properly parsed (#14532)
- Resolved `LightningApp(..., debug=True)` (#14464)


## [0.6.0] - 2022-09-08

### Added

- Introduce lightning connect ([#14452](https://github.com/Lightning-AI/lightning/pull/14452))
- Adds `PanelFrontend` to easily create complex UI in Python ([#13531](https://github.com/Lightning-AI/lightning/pull/13531))
- Add support for `Lightning App Commands` through the `configure_commands` hook on the Lightning Flow and the `ClientCommand`  ([#13602](https://github.com/Lightning-AI/lightning/pull/13602))
- Add support for Lightning AI BYOC cluster management ([#13835](https://github.com/Lightning-AI/lightning/pull/13835))
- Add support to see Lightning AI BYOC cluster logs ([#14334](https://github.com/Lightning-AI/lightning/pull/14334))
- Add support to run Lightning apps on Lightning AI BYOC clusters ([#13894](https://github.com/Lightning-AI/lightning/pull/13894))
- Add support for listing Lightning AI apps ([#13987](https://github.com/Lightning-AI/lightning/pull/13987))
- Adds `LightningTrainerScript`. `LightningTrainerScript` orchestrates multi-node training in the cloud ([#13830](https://github.com/Lightning-AI/lightning/pull/13830))
- Add support for printing application logs using CLI `lightning show logs <app_name> [components]` ([#13634](https://github.com/Lightning-AI/lightning/pull/13634))
- Add support for `Lightning API` through the `configure_api` hook on the Lightning Flow and the `Post`, `Get`, `Delete`, `Put` HttpMethods ([#13945](https://github.com/Lightning-AI/lightning/pull/13945))
- Added a warning when `configure_layout` returns URLs configured with http instead of https ([#14233](https://github.com/Lightning-AI/lightning/pull/14233))
- Add `--app_args` support from the CLI ([#13625](https://github.com/Lightning-AI/lightning/pull/13625))

### Changed

- Default values and parameter names for Lightning AI BYOC cluster management ([#14132](https://github.com/Lightning-AI/lightning/pull/14132))
- Run the flow only if the state has changed from the previous execution ([#14076](https://github.com/Lightning-AI/lightning/pull/14076))
- Increased DeepDiff's verbose level to properly handle dict changes ([#13960](https://github.com/Lightning-AI/lightning/pull/13960))
- Setup: added requirement freeze for next major version ([#14480](https://github.com/Lightning-AI/lightning/pull/14480))

### Fixed

- Unification of app template: moved `app.py` to root dir for `lightning init app <app_name>` template ([#13853](https://github.com/Lightning-AI/lightning/pull/13853))
- Fixed an issue with `lightning --version` command ([#14433](https://github.com/Lightning-AI/lightning/pull/14433))
- Fixed imports of collections.abc for py3.10 ([#14345](https://github.com/Lightning-AI/lightning/pull/14345))

## [0.5.7] - 2022-08-22

### Changed

- Release LAI docs as stable ([#14250](https://github.com/Lightning-AI/lightning/pull/14250))
- Compatibility for Python 3.10

### Fixed

- Pinning starsessions to 1.x ([#14333](https://github.com/Lightning-AI/lightning/pull/14333))
- Parsed local package versions ([#13933](https://github.com/Lightning-AI/lightning/pull/13933))


## [0.5.6] - 2022-08-16

### Fixed

- Resolved a bug where the `install` command was not installing the latest version of an app/component by default ([#14181](https://github.com/Lightning-AI/lightning/pull/14181))


## [0.5.5] - 2022-08-9

### Deprecated

- Deprecate sheety API ([#14004](https://github.com/Lightning-AI/lightning/pull/14004))

### Fixed

- Resolved a bug where the work statuses will grow quickly and be duplicated ([#13970](https://github.com/Lightning-AI/lightning/pull/13970))
- Resolved a bug about a race condition when sending the work state through the caller_queue ([#14074](https://github.com/Lightning-AI/lightning/pull/14074))
- Fixed Start Lightning App on Cloud if Repo Begins With Name "Lightning" ([#14025](https://github.com/Lightning-AI/lightning/pull/14025))


## [0.5.4] - 2022-08-01

### Changed

- Wrapped imports for traceability ([#13924](https://github.com/Lightning-AI/lightning/pull/13924))
- Set version as today ([#13906](https://github.com/Lightning-AI/lightning/pull/13906))

### Fixed

- Included app templates to the lightning and app packages ([#13731](https://github.com/Lightning-AI/lightning/pull/13731))
- Added UI for install all ([#13732](https://github.com/Lightning-AI/lightning/pull/13732))
- Fixed build meta pkg flow ([#13926](https://github.com/Lightning-AI/lightning/pull/13926))

## [0.5.3] - 2022-07-25

### Changed

- Pruned requirements duplicity ([#13739](https://github.com/Lightning-AI/lightning/pull/13739))

### Fixed

- Use correct python version in lightning component template ([#13790](https://github.com/Lightning-AI/lightning/pull/13790))

## [0.5.2] - 2022-07-18

### Added

- Update the Lightning App docs ([#13537](https://github.com/Lightning-AI/lightning/pull/13537))

### Changed

- Added `LIGHTNING_` prefix to Platform AWS credentials ([#13703](https://github.com/Lightning-AI/lightning/pull/13703))
