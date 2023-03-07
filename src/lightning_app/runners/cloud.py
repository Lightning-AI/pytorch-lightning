# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import json
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import click
import rich
from lightning_cloud.openapi import (
    Body3,
    Body4,
    CloudspaceIdRunsBody,
    Externalv1LightningappInstance,
    Gridv1ImageSpec,
    IdGetBody1,
    ProjectIdCloudspacesBody,
    V1BuildSpec,
    V1CloudSpace,
    V1DependencyFileInfo,
    V1Drive,
    V1DriveSpec,
    V1DriveStatus,
    V1DriveType,
    V1EnvVar,
    V1Flowserver,
    V1LightningappInstanceSpec,
    V1LightningappInstanceState,
    V1LightningAuth,
    V1LightningBasicAuth,
    V1LightningRun,
    V1LightningworkDrives,
    V1LightningworkSpec,
    V1Membership,
    V1Metadata,
    V1NetworkConfig,
    V1PackageManager,
    V1PythonDependencyInfo,
    V1QueueServerType,
    V1SourceType,
    V1UserRequestedComputeConfig,
    V1UserRequestedFlowComputeConfig,
    V1Work,
)
from lightning_cloud.openapi.rest import ApiException

from lightning_app.core.app import LightningApp
from lightning_app.core.constants import (
    CLOUD_UPLOAD_WARNING,
    DEFAULT_NUMBER_OF_EXPOSED_PORTS,
    DISABLE_DEPENDENCY_CACHE,
    ENABLE_APP_COMMENT_COMMAND_EXECUTION,
    enable_interruptible_works,
    enable_multiple_works_in_default_container,
    ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER,
    ENABLE_PULLING_STATE_ENDPOINT,
    ENABLE_PUSHING_STATE_ENDPOINT,
    get_cloud_queue_type,
    get_cluster_driver,
    get_lightning_cloud_url,
    LIGHTNING_CLOUD_PRINT_SPECS,
)
from lightning_app.core.work import LightningWork
from lightning_app.runners.backends.cloud import CloudBackend
from lightning_app.runners.runtime import Runtime
from lightning_app.source_code import LocalSourceCodeDir
from lightning_app.source_code.copytree import _filter_ignored, _IGNORE_FUNCTION, _parse_lightningignore
from lightning_app.storage import Drive, Mount
from lightning_app.utilities.app_helpers import _is_headless, Logger
from lightning_app.utilities.auth import _credential_string_to_basic_auth_params
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.clusters import _ensure_cluster_project_binding, _get_default_cluster
from lightning_app.utilities.dependency_caching import get_hash
from lightning_app.utilities.load_app import load_app_from_file
from lightning_app.utilities.packaging.app_config import _get_config_file, AppConfig
from lightning_app.utilities.packaging.lightning_utils import _prepare_lightning_wheels_and_requirements
from lightning_app.utilities.secrets import _names_to_ids

logger = Logger(__name__)


def _to_clean_dict(swagger_object, map_attributes):
    """Returns the swagger object properties as a dict with correct object names."""

    if hasattr(swagger_object, "to_dict"):
        attribute_map = swagger_object.attribute_map
        result = {}
        for key in attribute_map.keys():
            value = getattr(swagger_object, key)
            value = _to_clean_dict(value, map_attributes)
            if value is not None and value != {}:
                key = attribute_map[key] if map_attributes else key
                result[key] = value
        return result
    elif isinstance(swagger_object, list):
        return [_to_clean_dict(x, map_attributes) for x in swagger_object]
    elif isinstance(swagger_object, dict):
        return {key: _to_clean_dict(value, map_attributes) for key, value in swagger_object.items()}
    return swagger_object


@dataclass
class CloudRuntime(Runtime):
    backend: Union[str, CloudBackend] = "cloud"

    def open(self, name: str, cluster_id: Optional[str] = None):
        """Method to open a CloudSpace with the root folder uploaded."""
        try:
            # Check for feature support
            user = self.backend.client.auth_service_get_user()
            if not user.features.code_tab:
                rich.print(
                    "[red]The `lightning open` command has not been enabled for your account. "
                    "To request access, please contact support@lightning.ai[/red]"
                )
                sys.exit(1)

            # Dispatch in four phases: resolution, validation, spec creation, API transactions
            # Resolution
            cloudspace_config = self._resolve_config(name, load=False)
            root = self._resolve_root()
            ignore_functions = self._resolve_open_ignore_functions()
            repo = self._resolve_repo(root, ignore_functions)
            project = self._resolve_project()
            existing_cloudspaces = self._resolve_existing_cloudspaces(project.project_id, cloudspace_config.name)
            cluster_id = self._resolve_cluster_id(cluster_id, project.project_id, existing_cloudspaces)
            existing_cloudspace, existing_run_instance = self._resolve_existing_run_instance(
                cluster_id, project.project_id, existing_cloudspaces
            )
            cloudspace_name = self._resolve_cloudspace_name(
                cloudspace_config.name,
                existing_cloudspace,
                existing_cloudspaces,
            )
            needs_credits = self._resolve_needs_credits(project)

            # Validation
            # Note: We do not validate the repo here since open only uploads a directory if asked explicitly
            self._validate_cluster_id(cluster_id, project.project_id)

            # Spec creation
            run_body = self._get_run_body(cluster_id, [], None, [], True, root, self.start_server)

            if existing_run_instance is not None:
                print(
                    f"Re-opening the CloudSpace {cloudspace_config.name}. "
                    "This operation will create a new run but will not overwrite the files in your CloudSpace."
                )
            else:
                print(f"The name of the CloudSpace is: {cloudspace_config.name}")

            # API transactions
            cloudspace_id = self._api_create_cloudspace_if_not_exists(
                project.project_id,
                cloudspace_name,
                existing_cloudspace,
            )
            self._api_stop_existing_run_instance(project.project_id, existing_run_instance)
            run = self._api_create_run(project.project_id, cloudspace_id, run_body)
            self._api_package_and_upload_repo(repo, run)

            if getattr(run, "cluster_id", None):
                print(f"Running on {run.cluster_id}")

            if "PYTEST_CURRENT_TEST" not in os.environ:
                click.launch(self._get_cloudspace_url(project, cloudspace_name, "code", needs_credits))

        except ApiException as e:
            logger.error(e.body)
            sys.exit(1)

    def cloudspace_dispatch(
        self,
        project_id: str,
        cloudspace_id: str,
        name: str,
        cluster_id: str,
    ) -> str:
        """Slim dispatch for creating runs from a cloudspace. This dispatch avoids resolution of some properties
        such as the project and cluster IDs that are instead passed directly.

        Args:
            project_id: The ID of the project.
            cloudspace_id: The ID of the cloudspace.
            name: The name for the run.
            cluster_id: The ID of the cluster to run on.

        Raises:
            ApiException: If there was an issue in the backend.
            RuntimeError: If there are validation errors.
            ValueError: If there are validation errors.

        Returns:
            The URL of the created job.
        """
        # Dispatch in four phases: resolution, validation, spec creation, API transactions
        # Resolution
        root = self._resolve_root()
        repo = self._resolve_repo(root)
        project = self._resolve_project(project_id=project_id)
        existing_instances = self._resolve_run_instances_by_name(project_id, name)
        name = self._resolve_run_name(name, existing_instances)
        queue_server_type = self._resolve_queue_server_type()

        self.app._update_index_file()

        # Validation
        # TODO: Validate repo and surface to the user
        # self._validate_repo(root, repo)
        self._validate_work_build_specs_and_compute()
        self._validate_drives()
        self._validate_mounts()

        # Spec creation
        flow_servers = self._get_flow_servers()
        network_configs = self._get_network_configs(flow_servers)
        works = self._get_works()
        run_body = self._get_run_body(cluster_id, flow_servers, network_configs, works, False, root, True)
        env_vars = self._get_env_vars(self.env_vars, self.secrets, self.run_app_comment_commands)

        # API transactions
        run = self._api_create_run(project_id, cloudspace_id, run_body)
        self._api_package_and_upload_repo(repo, run)

        run_instance = self._api_create_run_instance(
            cluster_id,
            project_id,
            name,
            cloudspace_id,
            run.id,
            V1LightningappInstanceState.RUNNING,
            queue_server_type,
            env_vars,
        )

        return self._get_app_url(project, run_instance, "logs" if run.is_headless else "web-ui")

    def dispatch(
        self,
        name: str = "",
        cluster_id: str = None,
        open_ui: bool = True,
        no_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        """Method to dispatch and run the :class:`~lightning_app.core.app.LightningApp` in the cloud."""
        # not user facing error ideally - this should never happen in normal user workflow
        if not self.entrypoint:
            raise ValueError(
                "Entrypoint file not provided. Did you forget to "
                "initialize the Runtime object with `entrypoint` argument?"
            )

        cleanup_handle = None

        try:
            # Dispatch in four phases: resolution, validation, spec creation, API transactions
            # Resolution
            cloudspace_config = self._resolve_config(name)
            root = self._resolve_root()
            repo = self._resolve_repo(root)
            project = self._resolve_project()
            existing_cloudspaces = self._resolve_existing_cloudspaces(project.project_id, cloudspace_config.name)
            cluster_id = self._resolve_cluster_id(cluster_id, project.project_id, existing_cloudspaces)
            existing_cloudspace, existing_run_instance = self._resolve_existing_run_instance(
                cluster_id, project.project_id, existing_cloudspaces
            )
            cloudspace_name = self._resolve_cloudspace_name(
                cloudspace_config.name,
                existing_cloudspace,
                existing_cloudspaces,
            )
            queue_server_type = self._resolve_queue_server_type()
            needs_credits = self._resolve_needs_credits(project)

            # TODO: Move these
            cleanup_handle = _prepare_lightning_wheels_and_requirements(root)
            self.app._update_index_file()

            # Validation
            self._validate_repo(root, repo)
            self._validate_cluster_id(cluster_id, project.project_id)
            self._validate_work_build_specs_and_compute()
            self._validate_drives()
            self._validate_mounts()

            # Spec creation
            flow_servers = self._get_flow_servers()
            network_configs = self._get_network_configs(flow_servers)
            works = self._get_works()
            run_body = self._get_run_body(
                cluster_id, flow_servers, network_configs, works, no_cache, root, self.start_server
            )
            auth = self._get_auth(self.enable_basic_auth)
            env_vars = self._get_env_vars(self.env_vars, self.secrets, self.run_app_comment_commands)

            if LIGHTNING_CLOUD_PRINT_SPECS is not None:
                self._print_specs(run_body, LIGHTNING_CLOUD_PRINT_SPECS)
                sys.exit(0)

            print(f"The name of the app is: {cloudspace_name}")

            # API transactions
            cloudspace_id = self._api_create_cloudspace_if_not_exists(
                project.project_id,
                cloudspace_name,
                existing_cloudspace,
            )
            self._api_stop_existing_run_instance(project.project_id, existing_run_instance)
            run = self._api_create_run(project.project_id, cloudspace_id, run_body)
            self._api_package_and_upload_repo(repo, run)

            if getattr(run, "cluster_id", None):
                print(f"Running app on {run.cluster_id}")

            # Save the config for re-runs
            cloudspace_config.save_to_dir(root)

            desired_state = (
                V1LightningappInstanceState.STOPPED if needs_credits else V1LightningappInstanceState.RUNNING
            )

            if existing_run_instance is not None:
                run_instance = self._api_transfer_run_instance(
                    project.project_id,
                    run.id,
                    existing_run_instance.id,
                    desired_state,
                    queue_server_type,
                    env_vars,
                    auth,
                )
            else:
                run_instance = self._api_create_run_instance(
                    cluster_id,
                    project.project_id,
                    cloudspace_name,
                    cloudspace_id,
                    run.id,
                    desired_state,
                    queue_server_type,
                    env_vars,
                    auth,
                )

            if run_instance.status.phase == V1LightningappInstanceState.FAILED:
                raise RuntimeError("Failed to create the application. Cannot upload the source code.")

            # TODO: Remove testing dependency, but this would open a tab for each test...
            if open_ui and "PYTEST_CURRENT_TEST" not in os.environ:
                click.launch(
                    self._get_app_url(project, run_instance, "logs" if run.is_headless else "web-ui", needs_credits)
                )

            if bool(int(os.getenv("LIGHTING_TESTING", "0"))):
                print(f"APP_LOGS_URL: {self._get_app_url(project, run_instance, 'logs')}")

        except ApiException as e:
            logger.error(e.body)
            sys.exit(1)
        finally:
            if cleanup_handle:
                cleanup_handle()

    @classmethod
    def load_app_from_file(cls, filepath: str) -> "LightningApp":
        """Load a LightningApp from a file, mocking the imports."""

        # Pretend we are running in the cloud when loading the app locally
        os.environ["LAI_RUNNING_IN_CLOUD"] = "1"

        try:
            app = load_app_from_file(filepath, raise_exception=True, mock_imports=True)
        except FileNotFoundError as e:
            raise e
        except Exception:
            from lightning_app.testing.helpers import EmptyFlow

            # Create a generic app.
            logger.info("Could not load the app locally. Starting the app directly on the cloud.")
            app = LightningApp(EmptyFlow())
        finally:
            del os.environ["LAI_RUNNING_IN_CLOUD"]
        return app

    def _resolve_config(self, name: Optional[str], load: bool = True) -> AppConfig:
        """Find and load the config file if it exists (otherwise create an empty config).

        Override the name if provided.
        """
        config_file = _get_config_file(self.entrypoint)
        cloudspace_config = AppConfig.load_from_file(config_file) if config_file.exists() and load else AppConfig()
        if name:
            # Override the name if provided
            cloudspace_config.name = name
        return cloudspace_config

    def _resolve_root(self) -> Path:
        """Determine the root of the project."""
        root = Path(self.entrypoint).absolute()
        if root.is_file():
            root = root.parent
        return root

    def _resolve_open_ignore_functions(self) -> List[_IGNORE_FUNCTION]:
        """Used by the ``open`` method.

        If the entrypoint is a file, return an ignore function that will ignore everything except that file so only the
        file gets uploaded.
        """
        entrypoint = self.entrypoint.absolute()
        if entrypoint.is_file():
            return [lambda src, paths: [path for path in paths if path.absolute() == entrypoint]]
        return []

    def _resolve_repo(
        self,
        root: Path,
        ignore_functions: Optional[List[_IGNORE_FUNCTION]] = None,
    ) -> LocalSourceCodeDir:
        """Gather and merge all lightningignores from the app children and create the ``LocalSourceCodeDir``
        object."""
        if ignore_functions is None:
            ignore_functions = []

        if self.app is not None:
            flow_lightningignores = [flow.lightningignore for flow in self.app.flows]
            work_lightningignores = [work.lightningignore for work in self.app.works]
            lightningignores = flow_lightningignores + work_lightningignores
            if lightningignores:
                merged = sum(lightningignores, tuple())
                logger.debug(f"Found the following lightningignores: {merged}")
                patterns = _parse_lightningignore(merged)
                ignore_functions = [*ignore_functions, partial(_filter_ignored, root, patterns)]

        return LocalSourceCodeDir(path=root, ignore_functions=ignore_functions)

    def _resolve_project(self, project_id: Optional[str] = None) -> V1Membership:
        """Determine the project to run on, choosing a default if multiple projects are found."""
        return _get_project(self.backend.client, project_id=project_id)

    def _resolve_existing_cloudspaces(self, project_id: str, cloudspace_name: str) -> List[V1CloudSpace]:
        """Lists all the cloudspaces with a name matching the provided cloudspace name."""
        # TODO: Add pagination, otherwise this could break if users have a lot of cloudspaces.
        existing_cloudspaces = self.backend.client.cloud_space_service_list_cloud_spaces(
            project_id=project_id
        ).cloudspaces

        # Search for cloudspaces with the given name (possibly with some random characters appended)
        pattern = re.escape(f"{cloudspace_name}-") + ".{4}"
        return [
            cloudspace
            for cloudspace in existing_cloudspaces
            if cloudspace.name == cloudspace_name or (re.fullmatch(pattern, cloudspace.name) is not None)
        ]

    def _resolve_cluster_id(
        self, cluster_id: Optional[str], project_id: str, existing_cloudspaces: List[V1CloudSpace]
    ) -> Optional[str]:
        """If cloudspaces exist and cluster is None, mimic cluster selection logic to choose a default."""
        if cluster_id is None and len(existing_cloudspaces) > 0:
            # Determine the cluster ID
            cluster_id = _get_default_cluster(self.backend.client, project_id)
        return cluster_id

    def _resolve_existing_run_instance(
        self, cluster_id: Optional[str], project_id: str, existing_cloudspaces: List[V1CloudSpace]
    ) -> Tuple[Optional[V1CloudSpace], Optional[Externalv1LightningappInstance]]:
        """Look for an existing run and instance from one of the provided cloudspaces on the provided cluster."""
        existing_cloudspace = None
        existing_run_instance = None

        if cluster_id is not None:
            for cloudspace in existing_cloudspaces:
                run_instances = self.backend.client.lightningapp_instance_service_list_lightningapp_instances(
                    project_id=project_id,
                    app_id=cloudspace.id,
                ).lightningapps
                if run_instances and run_instances[0].spec.cluster_id == cluster_id:
                    existing_cloudspace = cloudspace
                    existing_run_instance = run_instances[0]
                    break
        return existing_cloudspace, existing_run_instance

    def _resolve_run_instances_by_name(self, project_id: str, name: str) -> List[Externalv1LightningappInstance]:
        """Get all existing instances in the given project with the given name."""
        run_instances = self.backend.client.lightningapp_instance_service_list_lightningapp_instances(
            project_id=project_id,
        ).lightningapps

        return [run_instance for run_instance in run_instances if run_instance.display_name == name]

    def _resolve_cloudspace_name(
        self,
        cloudspace_name: str,
        existing_cloudspace: Optional[V1CloudSpace],
        existing_cloudspaces: List[V1CloudSpace],
    ) -> str:
        """If there are existing cloudspaces but not on the cluster - choose a randomised name."""
        if len(existing_cloudspaces) > 0 and existing_cloudspace is None:
            name_exists = True
            while name_exists:
                random_name = cloudspace_name + "-" + "".join(random.sample(string.ascii_letters, 4))
                name_exists = any([app.name == random_name for app in existing_cloudspaces])

            cloudspace_name = random_name
        return cloudspace_name

    def _resolve_run_name(
        self,
        name: str,
        existing_instances: List[Externalv1LightningappInstance],
    ) -> str:
        """If there are existing instances with the same name - choose a randomised name."""
        if len(existing_instances) > 0:
            name_exists = True
            while name_exists:
                random_name = name + "-" + "".join(random.sample(string.ascii_letters, 4))
                name_exists = any([app.name == random_name for app in existing_instances])

            name = random_name
        return name

    def _resolve_queue_server_type(self) -> V1QueueServerType:
        """Resolve the cloud queue type from the environment."""
        queue_server_type = V1QueueServerType.UNSPECIFIED
        # Note: Enable app to select their own queue type.
        queue_type = get_cloud_queue_type()
        if queue_type == "http":
            queue_server_type = V1QueueServerType.HTTP
        elif queue_type == "redis":
            queue_server_type = V1QueueServerType.REDIS
        return queue_server_type

    @staticmethod
    def _resolve_needs_credits(project: V1Membership):
        """Check if the user likely needs credits to run the app with its hardware.

        Returns False if user has 1 or more credits.
        """
        balance = project.balance
        if balance is None:
            balance = 0  # value is missing in some tests

        needs_credits = balance < 1
        if needs_credits:
            logger.warn("You may need Lightning credits to run your apps on the cloud.")
        return needs_credits

    @staticmethod
    def _validate_repo(root: Path, repo: LocalSourceCodeDir) -> None:
        """This method is used to inform the users if their folder files are large and how to filter them."""
        excludes = set(fnmatch.filter(repo.files, "*lightning-*.tar.gz"))
        excludes.update(fnmatch.filter(repo.files, ".lightningignore"))
        files = [Path(f) for f in repo.files if f not in excludes]
        file_sizes = {f: f.stat().st_size for f in files}
        mb = 1000_000
        app_folder_size_in_mb = sum(file_sizes.values()) / mb
        if app_folder_size_in_mb > CLOUD_UPLOAD_WARNING:
            # filter out files under 0.01mb
            relevant_files = {f: sz for f, sz in file_sizes.items() if sz > 0.01 * mb}
            if relevant_files:
                by_largest = dict(sorted(relevant_files.items(), key=lambda x: x[1], reverse=True))
                by_largest = dict(list(by_largest.items())[:25])  # trim
                largest_paths_msg = "\n".join(
                    f"{round(sz / mb, 5)} MB: {p.relative_to(root)}" for p, sz in by_largest.items()
                )
                largest_paths_msg = f"Here are the largest files:\n{largest_paths_msg}\n"
            else:
                largest_paths_msg = ""
            warning_msg = (
                f"Your application folder '{root.absolute()}' is more than {CLOUD_UPLOAD_WARNING} MB. "
                f"The total size is {round(app_folder_size_in_mb, 2)} MB. {len(files)} files were uploaded.\n"
                + largest_paths_msg
                + "Perhaps you should try running the app in an empty directory.\n"
                + "You can ignore some files or folders by adding them to `.lightningignore`.\n"
                + " You can also set the `self.lightningingore` attribute in a Flow or Work."
            )

            logger.warn(warning_msg)

    def _validate_cluster_id(self, cluster_id: Optional[str], project_id: str):
        """Check that the provided cluster exists and ensure that it is bound to the given project."""
        if cluster_id is not None:
            # Verify that the cluster exists
            list_clusters_resp = self.backend.client.cluster_service_list_clusters()
            cluster_ids = [cluster.id for cluster in list_clusters_resp.clusters]
            if cluster_id not in cluster_ids:
                raise ValueError(f"You requested to run on cluster {cluster_id}, but that cluster doesn't exist.")

            _ensure_cluster_project_binding(self.backend.client, project_id, cluster_id)

    def _validate_work_build_specs_and_compute(self) -> None:
        """Check that the cloud compute and build configs are valid for all works in the app."""
        for work in self.app.works:
            if work.cloud_build_config.image is not None and work.cloud_compute.name == "default":
                raise ValueError(
                    f"You requested a custom base image for the Work with name '{work.name}', but custom images are "
                    "currently not supported on the default cloud compute instance. Please choose a different "
                    "configuration, for example `CloudCompute('cpu-medium')`."
                )

    def _validate_drives(self) -> None:
        """Check that all drives in the app have a valid protocol."""
        for work in self.app.works:
            for drive_attr_name, drive in [
                (k, getattr(work, k)) for k in work._state if isinstance(getattr(work, k), Drive)
            ]:
                if drive.protocol != "lit://":
                    raise RuntimeError(
                        f"Unknown drive protocol `{drive.protocol}` for drive `{work.name}.{drive_attr_name}`."
                    )

    def _validate_mounts(self) -> None:
        """Check that all mounts in the app have a valid protocol."""
        for work in self.app.works:
            if work.cloud_compute.mounts is not None:
                mounts = work.cloud_compute.mounts
                for mount in [mounts] if isinstance(mounts, Mount) else mounts:
                    if mount.protocol != "s3://":
                        raise RuntimeError(f"Unknown mount protocol `{mount.protocol}` for work `{work.name}`.")

    def _get_flow_servers(self) -> List[V1Flowserver]:
        """Collect a spec for each flow that contains a frontend so that the backend knows for which flows it needs
        to start servers."""
        flow_servers: List[V1Flowserver] = []
        for flow_name in self.app.frontends.keys():
            flow_server = V1Flowserver(name=flow_name)
            flow_servers.append(flow_server)
        return flow_servers

    @staticmethod
    def _get_network_configs(flow_servers: List[V1Flowserver]) -> Optional[List[V1NetworkConfig]]:
        """Get the list of network configs for the run if multiple works in default container is enabled."""
        network_configs = None
        if enable_multiple_works_in_default_container():
            network_configs = []
            initial_port = 8080 + 1 + len(flow_servers)
            for _ in range(DEFAULT_NUMBER_OF_EXPOSED_PORTS):
                network_configs.append(
                    V1NetworkConfig(
                        name="w" + str(initial_port),
                        port=initial_port,
                    )
                )
                initial_port += 1
        return network_configs

    @staticmethod
    def _get_drives(work: LightningWork) -> List[V1LightningworkDrives]:
        """Get the list of drive specifications for the provided work."""
        drives: List[V1LightningworkDrives] = []
        for drive_attr_name, drive in [
            (k, getattr(work, k)) for k in work._state if isinstance(getattr(work, k), Drive)
        ]:
            drives.append(
                V1LightningworkDrives(
                    drive=V1Drive(
                        metadata=V1Metadata(
                            name=f"{work.name}.{drive_attr_name}",
                        ),
                        spec=V1DriveSpec(
                            drive_type=V1DriveType.NO_MOUNT_S3,
                            source_type=V1SourceType.S3,
                            source=f"{drive.protocol}{drive.id}",
                        ),
                        status=V1DriveStatus(),
                    ),
                ),
            )

        return drives

    @staticmethod
    def _get_mounts(work: LightningWork) -> List[V1LightningworkDrives]:
        """Get the list of mount specifications for the provided work."""
        mounts = []
        if work.cloud_compute.mounts is not None:
            mount_objects = work.cloud_compute.mounts
            for mount in [mount_objects] if isinstance(mount_objects, Mount) else mount_objects:
                mounts.append(
                    V1LightningworkDrives(
                        drive=V1Drive(
                            metadata=V1Metadata(
                                name=work.name,
                            ),
                            spec=V1DriveSpec(
                                drive_type=V1DriveType.INDEXED_S3,
                                source_type=V1SourceType.S3,
                                source=mount.source,
                            ),
                            status=V1DriveStatus(),
                        ),
                        mount_location=str(mount.mount_path),
                    )
                )
        return mounts

    def _get_works(self) -> List[V1Work]:
        """Get the list of work specs from the app."""
        works: List[V1Work] = []
        for work in self.app.works:
            if not work._start_with_flow:
                continue

            work_requirements = "\n".join(work.cloud_build_config.requirements)
            build_spec = V1BuildSpec(
                commands=work.cloud_build_config.build_commands(),
                python_dependencies=V1PythonDependencyInfo(
                    package_manager=V1PackageManager.PIP, packages=work_requirements
                ),
                image=work.cloud_build_config.image,
            )
            user_compute_config = V1UserRequestedComputeConfig(
                name=work.cloud_compute.name,
                count=1,
                disk_size=work.cloud_compute.disk_size,
                preemptible=work.cloud_compute.interruptible,
                shm_size=work.cloud_compute.shm_size,
            )

            drives = self._get_drives(work)
            mounts = self._get_mounts(work)

            random_name = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
            work_spec = V1LightningworkSpec(
                build_spec=build_spec,
                drives=drives + mounts,
                user_requested_compute_config=user_compute_config,
                network_config=[V1NetworkConfig(name=random_name, port=work.port)],
            )
            works.append(V1Work(name=work.name, spec=work_spec))

        return works

    def _get_run_body(
        self,
        cluster_id: str,
        flow_servers: List[V1Flowserver],
        network_configs: Optional[List[V1NetworkConfig]],
        works: List[V1Work],
        no_cache: bool,
        root: Path,
        start_server: bool,
    ) -> CloudspaceIdRunsBody:
        """Get the specification of the run creation request."""
        # The entry point file needs to be relative to the root of the uploaded source file directory,
        # because the backend will invoke the lightning commands relative said source directory
        # TODO: we shouldn't set this if the entrypoint isn't a file but the backend gives an error if we don't
        app_entrypoint_file = Path(self.entrypoint).absolute().relative_to(root)

        run_body = CloudspaceIdRunsBody(
            cluster_id=cluster_id,
            app_entrypoint_file=str(app_entrypoint_file),
            enable_app_server=start_server,
            flow_servers=flow_servers,
            network_config=network_configs,
            works=works,
            local_source=True,
        )

        if self.app is not None:
            run_body.user_requested_flow_compute_config = V1UserRequestedFlowComputeConfig(
                name=self.app.flow_cloud_compute.name,
                shm_size=self.app.flow_cloud_compute.shm_size,
                preemptible=False,
            )

            run_body.is_headless = _is_headless(self.app)

        # if requirements file at the root of the repository is present,
        # we pass just the file name to the backend, so backend can find it in the relative path
        requirements_file = root / "requirements.txt"
        if requirements_file.is_file():
            run_body.image_spec = Gridv1ImageSpec(
                dependency_file_info=V1DependencyFileInfo(package_manager=V1PackageManager.PIP, path="requirements.txt")
            )
            if not DISABLE_DEPENDENCY_CACHE and not no_cache:
                # hash used for caching the dependencies
                run_body.dependency_cache_key = get_hash(requirements_file)

        return run_body

    @staticmethod
    def _get_auth(credentials: str) -> Optional[V1LightningAuth]:
        """If credentials are provided, parse them and return the auth spec."""
        auth = None
        if credentials != "":
            parsed_credentials = _credential_string_to_basic_auth_params(credentials)
            auth = V1LightningAuth(
                basic=V1LightningBasicAuth(
                    username=parsed_credentials["username"], password=parsed_credentials["password"]
                )
            )
        return auth

    @staticmethod
    def _get_env_vars(
        env_vars: Dict[str, str], secrets: Dict[str, str], run_app_comment_commands: bool
    ) -> List[V1EnvVar]:
        """Generate the list of environment variable specs for the app, including variables set by the
        framework."""
        v1_env_vars = [V1EnvVar(name=k, value=v) for k, v in env_vars.items()]

        if len(secrets.values()) > 0:
            secret_names_to_ids = _names_to_ids(secrets.values())
            env_vars_from_secrets = [V1EnvVar(name=k, from_secret=secret_names_to_ids[v]) for k, v in secrets.items()]
            v1_env_vars.extend(env_vars_from_secrets)

        if run_app_comment_commands or ENABLE_APP_COMMENT_COMMAND_EXECUTION:
            v1_env_vars.append(V1EnvVar(name="ENABLE_APP_COMMENT_COMMAND_EXECUTION", value="1"))

        if enable_multiple_works_in_default_container():
            v1_env_vars.append(V1EnvVar(name="ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", value="1"))

        if ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER:
            v1_env_vars.append(V1EnvVar(name="ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER", value="1"))

        if not ENABLE_PULLING_STATE_ENDPOINT:
            v1_env_vars.append(V1EnvVar(name="ENABLE_PULLING_STATE_ENDPOINT", value="0"))

        if not ENABLE_PUSHING_STATE_ENDPOINT:
            v1_env_vars.append(V1EnvVar(name="ENABLE_PUSHING_STATE_ENDPOINT", value="0"))

        if get_cloud_queue_type():
            v1_env_vars.append(V1EnvVar(name="LIGHTNING_CLOUD_QUEUE_TYPE", value=get_cloud_queue_type()))

        if get_cluster_driver():
            v1_env_vars.append(V1EnvVar(name="LIGHTNING_CLUSTER_DRIVER", value=get_cluster_driver()))

        if enable_interruptible_works():
            v1_env_vars.append(
                V1EnvVar(
                    name="LIGHTNING_INTERRUPTIBLE_WORKS",
                    value=os.getenv("LIGHTNING_INTERRUPTIBLE_WORKS", "0"),
                )
            )

        return v1_env_vars

    def _api_create_cloudspace_if_not_exists(
        self, project_id: str, name: str, existing_cloudspace: Optional[V1CloudSpace]
    ) -> str:
        """Create the cloudspace if it doesn't exist.

        Return the cloudspace ID.
        """
        if existing_cloudspace is None:
            cloudspace_body = ProjectIdCloudspacesBody(name=name, can_download_source_code=True)
            cloudspace = self.backend.client.cloud_space_service_create_cloud_space(
                project_id=project_id, body=cloudspace_body
            )
            return cloudspace.id
        return existing_cloudspace.id

    def _api_stop_existing_run_instance(
        self, project_id: str, existing_run_instance: Optional[Externalv1LightningappInstance]
    ) -> None:
        """If an existing instance is provided and it isn't stopped, stop it."""
        if existing_run_instance and existing_run_instance.status.phase != V1LightningappInstanceState.STOPPED:
            # TODO(yurij): Implement release switching in the UI and remove this
            # We can only switch release of the stopped instance
            existing_run_instance = self.backend.client.lightningapp_instance_service_update_lightningapp_instance(
                project_id=project_id,
                id=existing_run_instance.id,
                body=Body3(spec=V1LightningappInstanceSpec(desired_state=V1LightningappInstanceState.STOPPED)),
            )
            # wait for the instance to stop for up to 150 seconds
            for _ in range(150):
                existing_run_instance = self.backend.client.lightningapp_instance_service_get_lightningapp_instance(
                    project_id=project_id, id=existing_run_instance.id
                )
                if existing_run_instance.status.phase == V1LightningappInstanceState.STOPPED:
                    break
                time.sleep(1)
            if existing_run_instance.status.phase != V1LightningappInstanceState.STOPPED:
                raise RuntimeError("Failed to stop the existing instance.")

    def _api_create_run(self, project_id: str, cloudspace_id: str, run_body: CloudspaceIdRunsBody) -> V1LightningRun:
        """Create and return the run."""
        return self.backend.client.cloud_space_service_create_lightning_run(
            project_id=project_id, cloudspace_id=cloudspace_id, body=run_body
        )

    def _api_transfer_run_instance(
        self,
        project_id: str,
        run_id: str,
        instance_id: str,
        desired_state: V1LightningappInstanceState,
        queue_server_type: Optional[V1QueueServerType] = None,
        env_vars: Optional[List[V1EnvVar]] = None,
        auth: Optional[V1LightningAuth] = None,
    ) -> Externalv1LightningappInstance:
        """Transfer an existing instance to the given run ID and update its specification.

        Return the instance.
        """
        run_instance = self.backend.client.lightningapp_instance_service_update_lightningapp_instance_release(
            project_id=project_id,
            id=instance_id,
            body=Body4(release_id=run_id),
        )

        self.backend.client.lightningapp_instance_service_update_lightningapp_instance(
            project_id=project_id,
            id=instance_id,
            body=Body3(
                spec=V1LightningappInstanceSpec(
                    desired_state=desired_state,
                    queue_server_type=queue_server_type,
                    env=env_vars,
                    auth=auth,
                )
            ),
        )

        return run_instance

    def _api_create_run_instance(
        self,
        cluster_id: str,
        project_id: str,
        run_name: str,
        cloudspace_id: str,
        run_id: str,
        desired_state: V1LightningappInstanceState,
        queue_server_type: Optional[V1QueueServerType] = None,
        env_vars: Optional[List[V1EnvVar]] = None,
        auth: Optional[V1LightningAuth] = None,
    ) -> Externalv1LightningappInstance:
        """Create a new instance of the given run with the given specification."""
        return self.backend.client.cloud_space_service_create_lightning_run_instance(
            project_id=project_id,
            cloudspace_id=cloudspace_id,
            id=run_id,
            body=IdGetBody1(
                cluster_id=cluster_id,
                name=run_name,
                desired_state=desired_state,
                queue_server_type=queue_server_type,
                env=env_vars,
                auth=auth,
            ),
        )

    @staticmethod
    def _api_package_and_upload_repo(repo: LocalSourceCodeDir, run: V1LightningRun) -> None:
        """Package and upload the provided local source code directory to the provided run."""
        if run.source_upload_url == "":
            raise RuntimeError("The source upload url is empty.")
        repo.package()
        repo.upload(url=run.source_upload_url)

    @staticmethod
    def _print_specs(run_body: CloudspaceIdRunsBody, print_format: str) -> None:
        """Print the given run body in either `web` or `gallery` format."""
        if print_format not in ("web", "gallery"):
            raise ValueError(
                f"`LIGHTNING_CLOUD_PRINT_SPECS` should be either `web` or `gallery`. You provided: {print_format}"
            )

        flow_servers_json = [{"Name": flow_server.name} for flow_server in run_body.flow_servers]
        logger.info(f"flow_servers: {flow_servers_json}")
        works_json = json.dumps(_to_clean_dict(run_body.works, print_format == "web"), separators=(",", ":"))
        logger.info(f"works: {works_json}")
        logger.info(f"entrypoint_file: {run_body.app_entrypoint_file}")
        requirements_path = getattr(getattr(run_body.image_spec, "dependency_file_info", ""), "path", "")
        logger.info(f"requirements_path: {requirements_path}")

    def _get_cloudspace_url(
        self, project: V1Membership, cloudspace_name: str, tab: str, need_credits: bool = False
    ) -> str:
        user = self.backend.client.auth_service_get_user()
        action = "?action=add_credits" if need_credits else ""
        paths = [
            user.username,
            project.name,
            "apps",
            cloudspace_name,
            tab,
        ]
        path = "/".join([quote(path, safe="") for path in paths])
        return f"{get_lightning_cloud_url()}/{path}{action}"

    def _get_app_url(
        self,
        project: V1Membership,
        run_instance: Externalv1LightningappInstance,
        tab: str,
        need_credits: bool = False,
    ) -> str:
        user = self.backend.client.auth_service_get_user()
        action = "?action=add_credits" if need_credits else ""
        if user.features.project_selector:
            paths = [
                user.username,
                project.name,
                "jobs",
                run_instance.name,
                tab,
            ]
        else:
            paths = [
                user.username,
                "apps",
                run_instance.id,
                tab,
            ]
        path = "/".join([quote(path, safe="") for path in paths])
        return f"{get_lightning_cloud_url()}/{path}{action}"
