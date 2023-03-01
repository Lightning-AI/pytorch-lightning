# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [1.9.4] - 2023-03-01

### Removed

- Removed implicit ui testing with `testing.run_app_in_cloud` in favor of headless login and app selection ([#16741](https://github.com/Lightning-AI/lightning/pull/16741))


## [1.9.3] - 2023-02-21

### Fixed

- Fixed `lightning open` command and improved redirects ([#16794](https://github.com/Lightning-AI/lightning/pull/16794))


## [1.9.2] - 2023-02-15

### Added

- Added Storage Commands ([#16740](https://github.com/Lightning-AI/lightning/pull/16740))
  * `rm`: Delete files from your Cloud Platform Filesystem
- Added `lightning connect data` to register data connection to private s3 buckets ([#16738](https://github.com/Lightning-AI/lightning/pull/16738))


## [1.9.1] - 2023-02-10

### Added

- Added `lightning open` command ([#16482](https://github.com/Lightning-AI/lightning/pull/16482))
- Added experimental support for interruptible GPU in the cloud ([#16399](https://github.com/Lightning-AI/lightning/pull/16399))
- Added FileSystem abstraction to simply manipulation of files ([#16581](https://github.com/Lightning-AI/lightning/pull/16581))
- Added Storage Commands ([#16606](https://github.com/Lightning-AI/lightning/pull/16606))
  * `ls`: List files from your Cloud Platform Filesystem
  * `cd`: Change the current directory within your Cloud Platform filesystem (terminal session based)
  * `pwd`: Return the current folder in your Cloud Platform Filesystem
  * `cp`: Copy files between your Cloud Platform Filesystem and local filesystem
- Prevent to `cd` into non existent folders ([#16645](https://github.com/Lightning-AI/lightning/pull/16645))
- Enabled `cp` (upload) at project level ([#16631](https://github.com/Lightning-AI/lightning/pull/16631))
- Enabled `ls` and `cp` (download) at project level ([#16622](https://github.com/Lightning-AI/lightning/pull/16622))
- Added `lightning connect data` to register data connection to s3 buckets ([#16670](https://github.com/Lightning-AI/lightning/pull/16670))
- Added support for running with multiprocessing in the cloud ([#16624](https://github.com/Lightning-AI/lightning/pull/16624))
- Initial plugin server ([#16523](https://github.com/Lightning-AI/lightning/pull/16523))
- Connect and Disconnect node ([#16700](https://github.com/Lightning-AI/lightning/pull/16700))

### Changed

- Changed the default `LightningClient(retry=False)` to `retry=True` ([#16382](https://github.com/Lightning-AI/lightning/pull/16382))
- Add support for async predict method in PythonServer and remove torch context ([#16453](https://github.com/Lightning-AI/lightning/pull/16453))
- Renamed `lightning.app.components.LiteMultiNode` to `lightning.app.components.FabricMultiNode` ([#16505](https://github.com/Lightning-AI/lightning/pull/16505))
- Changed the command `lightning connect` to `lightning connect app` for consistency ([#16670](https://github.com/Lightning-AI/lightning/pull/16670))
- Refactor cloud dispatch and update to new API ([#16456](https://github.com/Lightning-AI/lightning/pull/16456))
- Updated app URLs to the latest format ([#16568](https://github.com/Lightning-AI/lightning/pull/16568))

### Fixed

- Fixed a deadlock causing apps not to exit properly when running locally ([#16623](https://github.com/Lightning-AI/lightning/pull/16623))
- Fixed the Drive root_folder not parsed properly ([#16454](https://github.com/Lightning-AI/lightning/pull/16454))
- Fixed malformed path when downloading files using `lightning cp` ([#16626](https://github.com/Lightning-AI/lightning/pull/16626))
- Fixed app name in URL ([#16575](https://github.com/Lightning-AI/lightning/pull/16575))


## [1.9.0] - 2023-01-17

### Added

- Added a possibility to set up basic authentication for Lightning apps ([#16105](https://github.com/Lightning-AI/lightning/pull/16105))


### Changed

- The LoadBalancer now uses internal ip + port instead of URL exposed ([#16119](https://github.com/Lightning-AI/lightning/pull/16119))
- Added support for logging in different trainer stages with  `DeviceStatsMonitor`
([#16002](https://github.com/Lightning-AI/lightning/pull/16002))
- Changed `lightning_app.components.serve.gradio` to  `lightning_app.components.serve.gradio_server` ([#16201](https://github.com/Lightning-AI/lightning/pull/16201))
- Made cluster creation/deletion async by default ([#16185](https://github.com/Lightning-AI/lightning/pull/16185))
- Expose `LightningFlow.stop` method to stop the flow similar to works ([##16378](https://github.com/Lightning-AI/lightning/pull/16378))

### Fixed

- Fixed not being able to run multiple lightning apps locally due to port collision ([#15819](https://github.com/Lightning-AI/lightning/pull/15819))
- Avoid `relpath` bug on Windows ([#16164](https://github.com/Lightning-AI/lightning/pull/16164))
- Avoid using the deprecated `LooseVersion` ([#16162](https://github.com/Lightning-AI/lightning/pull/16162))
- Porting fixes to autoscaler component ([#16249](https://github.com/Lightning-AI/lightning/pull/16249))
- Fixed a bug where `lightning login` with env variables would not correctly save the credentials ([#16339](https://github.com/Lightning-AI/lightning/pull/16339))


## [1.8.6] - 2022-12-21

### Added

- Added partial support for fastapi `Request` annotation in `configure_api` handlers ([#16047](https://github.com/Lightning-AI/lightning/pull/16047))
- Added a nicer UI with URL and examples for the autoscaler component ([#16063](https://github.com/Lightning-AI/lightning/pull/16063))
- Enabled users to have more control over scaling out/in interval ([#16093](https://github.com/Lightning-AI/lightning/pull/16093))
- Added more datatypes to serving component ([#16018](https://github.com/Lightning-AI/lightning/pull/16018))
- Added `work.delete` method to delete the work ([#16103](https://github.com/Lightning-AI/lightning/pull/16103))
- Added `display_name` property to LightningWork for the cloud ([#16095](https://github.com/Lightning-AI/lightning/pull/16095))
- Added `ColdStartProxy` to the AutoScaler ([#16094](https://github.com/Lightning-AI/lightning/pull/16094))
- Added status endpoint, enable `ready` ([#16075](https://github.com/Lightning-AI/lightning/pull/16075))
- Implemented `ready` for components ([#16129](https://github.com/Lightning-AI/lightning/pull/16129))

### Changed

- The default `start_method` for creating Work processes locally on MacOS is now 'spawn' (previously 'fork') ([#16089](https://github.com/Lightning-AI/lightning/pull/16089))
- The utility `lightning.app.utilities.cloud.is_running_in_cloud` now returns `True` during loading of the app locally when running with `--cloud` ([#16045](https://github.com/Lightning-AI/lightning/pull/16045))
- Updated Multinode Warning ([#16091](https://github.com/Lightning-AI/lightning/pull/16091))
- Updated app testing ([#16000](https://github.com/Lightning-AI/lightning/pull/16000))
- Changed overwrite to `True` ([#16009](https://github.com/Lightning-AI/lightning/pull/16009))
- Simplified messaging in cloud dispatch ([#16160](https://github.com/Lightning-AI/lightning/pull/16160))
- Added annotations endpoint ([#16159](https://github.com/Lightning-AI/lightning/pull/16159))

### Fixed

- Fixed `PythonServer` messaging "Your app has started" ([#15989](https://github.com/Lightning-AI/lightning/pull/15989))
- Fixed auto-batching to enable batching for requests coming even after batch interval but is in the queue ([#16110](https://github.com/Lightning-AI/lightning/pull/16110))
- Fixed a bug where `AutoScaler` would fail with min_replica=0 ([#16092](https://github.com/Lightning-AI/lightning/pull/16092)
- Fixed a non-thread safe deepcopy in the scheduler ([#16114](https://github.com/Lightning-AI/lightning/pull/16114))
- Fixed Http Queue sleeping for 1 sec by default if no delta were found ([#16114](https://github.com/Lightning-AI/lightning/pull/16114))
- Fixed the endpoint info tab not showing up in `AutoScaler` UI ([#16128](https://github.com/Lightning-AI/lightning/pull/16128))
- Fixed an issue where an exception would be raised in the logs when using a recent version of streamlit ([#16139](https://github.com/Lightning-AI/lightning/pull/16139))
- Fixed e2e tests ([#16146](https://github.com/Lightning-AI/lightning/pull/16146))


## [1.8.5] - 2022-12-15

### Added

- Added `Lightning{Flow,Work}.lightningignores` attributes to programmatically ignore files before uploading to the cloud ([#15818](https://github.com/Lightning-AI/lightning/pull/15818))
- Added a progres bar while connecting to an app through the CLI ([#16035](https://github.com/Lightning-AI/lightning/pull/16035))
- Support running on multiple clusters ([#16016](https://github.com/Lightning-AI/lightning/pull/16016))
- Added guards to cluster deletion from cli ([#16053](https://github.com/Lightning-AI/lightning/pull/16053))

### Changed

- Cleanup cluster waiting ([#16054](https://github.com/Lightning-AI/lightning/pull/16054))

### Fixed

- Fixed `DDPStrategy` import in app framework ([#16029](https://github.com/Lightning-AI/lightning/pull/16029))
- Fixed `AutoScaler` raising an exception when non-default cloud compute is specified ([#15991](https://github.com/Lightning-AI/lightning/pull/15991))
- Fixed and improvements of login flow ([#16052](https://github.com/Lightning-AI/lightning/pull/16052))
- Fixed the debugger detection mechanism for lightning App in VSCode ([#16068](https://github.com/Lightning-AI/lightning/pull/16068))
- Fixed bug where components that are re-instantiated several times failed to initialize if they were modifying `self.lightningignore` ([#16080](https://github.com/Lightning-AI/lightning/pull/16080))
- Fixed a bug where apps that had previously been deleted could not be run again from the CLI ([#16082](https://github.com/Lightning-AI/lightning/pull/16082))
- Fixed install/upgrade - removing single quote ([#16079](https://github.com/Lightning-AI/lightning/pull/16079))


## [1.8.4] - 2022-12-08

### Added

- Add `code_dir` argument to tracer run ([#15771](https://github.com/Lightning-AI/lightning/pull/15771))
- Added the CLI command `lightning run model` to launch a `LightningLite` accelerated script ([#15506](https://github.com/Lightning-AI/lightning/pull/15506))
- Added the CLI command `lightning delete app` to delete a lightning app on the cloud ([#15783](https://github.com/Lightning-AI/lightning/pull/15783))
- Added a CloudMultiProcessBackend which enables running a child App from within the Flow in the cloud ([#15800](https://github.com/Lightning-AI/lightning/pull/15800))
- Utility for pickling work object safely even from a child process ([#15836](https://github.com/Lightning-AI/lightning/pull/15836))
- Added `AutoScaler` component (
   [#15769](https://github.com/Lightning-AI/lightning/pull/15769),
   [#15971](https://github.com/Lightning-AI/lightning/pull/15971),
   [#15966](https://github.com/Lightning-AI/lightning/pull/15966)
)
- Added the property `ready` of the LightningFlow to inform when the `Open App` should be visible ([#15921](https://github.com/Lightning-AI/lightning/pull/15921))
- Added private work attributed `_start_method` to customize how to start the works ([#15923](https://github.com/Lightning-AI/lightning/pull/15923))
- Added a `configure_layout` method to the `LightningWork` which can be used to control how the work is handled in the layout of a parent flow ([#15926](https://github.com/Lightning-AI/lightning/pull/15926))
- Added the ability to run a Lightning App or Component directly from the Gallery using `lightning run app organization/name` ([#15941](https://github.com/Lightning-AI/lightning/pull/15941))
- Added automatic conversion of list and dict of works and flows to structures ([#15961](https://github.com/Lightning-AI/lightning/pull/15961))

### Changed

- The `MultiNode` components now warn the user when running with `num_nodes > 1` locally ([#15806](https://github.com/Lightning-AI/lightning/pull/15806))
- Cluster creation and deletion now waits by default [#15458](https://github.com/Lightning-AI/lightning/pull/15458)
- Running an app without a UI locally no longer opens the browser ([#15875](https://github.com/Lightning-AI/lightning/pull/15875))
- Show a message when `BuildConfig(requirements=[...])` is passed but a `requirements.txt` file is already present in the Work ([#15799](https://github.com/Lightning-AI/lightning/pull/15799))
- Show a message when `BuildConfig(dockerfile="...")` is passed but a `Dockerfile` file is already present in the Work ([#15799](https://github.com/Lightning-AI/lightning/pull/15799))
- Dropped name column from cluster list ([#15721](https://github.com/Lightning-AI/lightning/pull/15721))
- Apps without UIs no longer activate the "Open App" button when running in the cloud ([#15875](https://github.com/Lightning-AI/lightning/pull/15875))
- Wait for full file to be transferred in Path / Payload ([#15934](https://github.com/Lightning-AI/lightning/pull/15934))

### Removed

- Removed the `SingleProcessRuntime` ([#15933](https://github.com/Lightning-AI/lightning/pull/15933))

### Fixed

- Fixed SSH CLI command listing stopped components ([#15810](https://github.com/Lightning-AI/lightning/pull/15810))
- Fixed bug when launching apps on multiple clusters ([#15484](https://github.com/Lightning-AI/lightning/pull/15484))
- Fixed Sigterm Handler causing thread lock which caused KeyboardInterrupt to hang ([#15881](https://github.com/Lightning-AI/lightning/pull/15881))
- Fixed MPS error for multinode component (defaults to cpu on mps devices now as distributed operations are not supported by pytorch on mps) ([#15748](https://github.com/Lightning-AI/lightning/pull/15748))
- Fixed the work not stopped when successful when passed directly to the LightningApp ([#15801](https://github.com/Lightning-AI/lightning/pull/15801))
- Fixed the PyTorch Inference locally on GPU ([#15813](https://github.com/Lightning-AI/lightning/pull/15813))
- Fixed the `enable_spawn` method of the `WorkRunExecutor` ([#15812](https://github.com/Lightning-AI/lightning/pull/15812))
- Fixed require/import decorator ([#15849](https://github.com/Lightning-AI/lightning/pull/15849))
- Fixed a bug where using `L.app.structures` would cause multiple apps to be opened and fail with an error in the cloud ([#15911](https://github.com/Lightning-AI/lightning/pull/15911))
- Fixed PythonServer generating noise on M1 ([#15949](https://github.com/Lightning-AI/lightning/pull/15949))
- Fixed multiprocessing breakpoint ([#15950](https://github.com/Lightning-AI/lightning/pull/15950))
- Fixed detection of a Lightning App running in debug mode ([#15951](https://github.com/Lightning-AI/lightning/pull/15951))
- Fixed `ImportError` on Multinode if package not present ([#15963](https://github.com/Lightning-AI/lightning/pull/15963))
- Fixed MultiNode Component to use separate cloud computes ([#15965](https://github.com/Lightning-AI/lightning/pull/15965))
- Fixed Registration for CloudComputes of Works in `L.app.structures` ([#15964](https://github.com/Lightning-AI/lightning/pull/15964))
- Fixed a bug where auto-upgrading to the latest lightning via the CLI could get stuck in a loop ([#15984](https://github.com/Lightning-AI/lightning/pull/15984))


## [1.8.3] - 2022-11-22

### Changed

- Deduplicate top level lighting CLI command groups ([#15761](https://github.com/Lightning-AI/lightning/pull/15761))
  * `lightning add ssh-key` CLI command has been transitioned to `lightning create ssh-key`
  * `lightning remove ssh-key` CLI command has been transitioned to `lightning delete ssh-key`
- Set Torch inference mode for prediction ([#15719](https://github.com/Lightning-AI/lightning/pull/15719))
- Improved `LightningTrainerScript` start-up time ([#15751](https://github.com/Lightning-AI/lightning/pull/15751))
- Disable XSRF protection in `StreamlitFrontend` to support upload in localhost ([#15684](https://github.com/Lightning-AI/lightning/pull/15684))

### Fixed

- Fixed debugging with VSCode IDE ([#15747](https://github.com/Lightning-AI/lightning/pull/15747))
- Fixed setting property to the `LightningFlow` ([#15750](https://github.com/Lightning-AI/lightning/pull/15750))
- Fixed the PyTorch Inference locally on GPU ([#15813](https://github.com/Lightning-AI/lightning/pull/15813))


## [1.8.2] - 2022-11-17

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

- Application storage prefix moved from `app_id` to `project_id/app_id` ([#14583](https://github.com/Lightning-AI/lightning/pull/14583))
- LightningCloud client calls to use keyword arguments instead of positional arguments ([#14685](https://github.com/Lightning-AI/lightning/pull/14685))

### Fixed

- Making `threadpool` non-default from LightningCloud client  ([#14757](https://github.com/Lightning-AI/lightning/pull/14757))
- Resolved a bug where the state change detection using DeepDiff won't work with Path, Drive objects ([#14465](https://github.com/Lightning-AI/lightning/pull/14465))
- Resolved a bug where the wrong client was passed to collect cloud logs ([#14684](https://github.com/Lightning-AI/lightning/pull/14684))
- Resolved the memory leak issue with the Lightning Cloud package and bumped the requirements to use the latest version ([#14697](https://github.com/Lightning-AI/lightning/pull/14697))
- Fixing 5000 log line limitation for Lightning AI BYOC cluster logs ([#14458](https://github.com/Lightning-AI/lightning/pull/14458))
- Fixed a bug where the uploaded command file wasn't properly parsed ([#14532](https://github.com/Lightning-AI/lightning/pull/14532))
- Resolved `LightningApp(..., debug=True)` ([#14464](https://github.com/Lightning-AI/lightning/pull/14464))


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


- Fixed the `examples/app_dag` example ([#14359](https://github.com/Lightning-AI/lightning/pull/14359))


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
