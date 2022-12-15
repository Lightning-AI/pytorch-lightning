import fnmatch
import json
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union

import click
from lightning_cloud.openapi import (
    Body3,
    Body4,
    Body7,
    Body8,
    Body9,
    Externalv1LightningappInstance,
    Gridv1ImageSpec,
    V1BuildSpec,
    V1ClusterType,
    V1DependencyFileInfo,
    V1Drive,
    V1DriveSpec,
    V1DriveStatus,
    V1DriveType,
    V1EnvVar,
    V1Flowserver,
    V1LightningappInstanceSpec,
    V1LightningappInstanceState,
    V1LightningworkDrives,
    V1LightningworkSpec,
    V1Membership,
    V1Metadata,
    V1NetworkConfig,
    V1PackageManager,
    V1ProjectClusterBinding,
    V1PythonDependencyInfo,
    V1QueueServerType,
    V1SourceType,
    V1UserRequestedComputeConfig,
    V1UserRequestedFlowComputeConfig,
    V1Work,
)
from lightning_cloud.openapi.rest import ApiException

from lightning_app import LightningWork
from lightning_app.core.app import LightningApp
from lightning_app.core.constants import (
    CLOUD_QUEUE_TYPE,
    CLOUD_UPLOAD_WARNING,
    DEFAULT_NUMBER_OF_EXPOSED_PORTS,
    DISABLE_DEPENDENCY_CACHE,
    DOT_IGNORE_FILENAME,
    ENABLE_APP_COMMENT_COMMAND_EXECUTION,
    enable_multiple_works_in_default_container,
    ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER,
    ENABLE_PULLING_STATE_ENDPOINT,
    ENABLE_PUSHING_STATE_ENDPOINT,
    get_lightning_cloud_url,
)
from lightning_app.runners.backends.cloud import CloudBackend
from lightning_app.runners.runtime import Runtime
from lightning_app.source_code import LocalSourceCodeDir
from lightning_app.source_code.copytree import _filter_ignored, _parse_lightningignore
from lightning_app.storage import Drive, Mount
from lightning_app.utilities.app_helpers import _is_headless, Logger
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.dependency_caching import get_hash
from lightning_app.utilities.load_app import load_app_from_file
from lightning_app.utilities.packaging.app_config import _get_config_file, AppConfig
from lightning_app.utilities.packaging.lightning_utils import _prepare_lightning_wheels_and_requirements
from lightning_app.utilities.secrets import _names_to_ids

logger = Logger(__name__)


def _get_work_specs(app: LightningApp) -> List[V1Work]:
    works: List[V1Work] = []
    for work in app.works:
        _validate_build_spec_and_compute(work)

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
            preemptible=work.cloud_compute.preemptible,
            shm_size=work.cloud_compute.shm_size,
        )

        drive_specs: List[V1LightningworkDrives] = []
        for drive_attr_name, drive in [
            (k, getattr(work, k)) for k in work._state if isinstance(getattr(work, k), Drive)
        ]:
            if drive.protocol == "lit://":
                drive_type = V1DriveType.NO_MOUNT_S3
                source_type = V1SourceType.S3
            else:
                raise RuntimeError(
                    f"unknown drive protocol `{drive.protocol}`. Please verify this "
                    f"drive type has been configured for use in the cloud dispatcher."
                )

            drive_specs.append(
                V1LightningworkDrives(
                    drive=V1Drive(
                        metadata=V1Metadata(
                            name=f"{work.name}.{drive_attr_name}",
                        ),
                        spec=V1DriveSpec(
                            drive_type=drive_type,
                            source_type=source_type,
                            source=f"{drive.protocol}{drive.id}",
                        ),
                        status=V1DriveStatus(),
                    ),
                ),
            )

        # TODO: Move this to the CloudCompute class and update backend
        if work.cloud_compute.mounts is not None:
            mounts = work.cloud_compute.mounts
            if isinstance(mounts, Mount):
                mounts = [mounts]
            for mount in mounts:
                drive_specs.append(
                    _create_mount_drive_spec(
                        work_name=work.name,
                        mount=mount,
                    )
                )

        random_name = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
        work_spec = V1LightningworkSpec(
            build_spec=build_spec,
            drives=drive_specs,
            user_requested_compute_config=user_compute_config,
            network_config=[V1NetworkConfig(name=random_name, port=work.port)],
        )
        works.append(V1Work(name=work.name, spec=work_spec))

    return works


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


def _generate_works_json(filepath: str, map_attributes: bool) -> str:
    app = CloudRuntime.load_app_from_file(filepath)
    works = _get_work_specs(app)
    works_json = json.dumps(_to_clean_dict(works, map_attributes), separators=(",", ":"))
    return works_json


def _generate_works_json_web(filepath: str) -> str:
    return _generate_works_json(filepath, True)


def _generate_works_json_gallery(filepath: str) -> str:
    return _generate_works_json(filepath, False)


@dataclass
class CloudRuntime(Runtime):

    backend: Union[str, CloudBackend] = "cloud"

    def dispatch(
        self,
        name: str = "",
        cluster_id: str = None,
        open_ui: bool = True,
        **kwargs: Any,
    ) -> None:
        """Method to dispatch and run the :class:`~lightning_app.core.app.LightningApp` in the cloud."""
        # not user facing error ideally - this should never happen in normal user workflow
        if not self.entrypoint_file:
            raise ValueError(
                "Entrypoint file not provided. Did you forget to "
                "initialize the Runtime object with `entrypoint_file` argument?"
            )

        # Determine the root of the project: Start at the entrypoint_file and look for nearby Lightning config files,
        # going up the directory structure. The root of the project is where the Lightning config file is located.

        config_file = _get_config_file(self.entrypoint_file)
        app_config = AppConfig.load_from_file(config_file) if config_file.exists() else AppConfig()
        root = Path(self.entrypoint_file).absolute().parent
        cleanup_handle = _prepare_lightning_wheels_and_requirements(root)
        self.app._update_index_file()

        # gather and merge all lightningignores
        children = self.app.flows + self.app.works
        lightningignores = [c.lightningignore for c in children]
        if lightningignores:
            merged = sum(lightningignores, tuple())
            logger.debug(f"Found the following lightningignores: {merged}")
            patterns = _parse_lightningignore(merged)
            ignore_functions = [partial(_filter_ignored, root, patterns)]
        else:
            ignore_functions = None

        # Create a default dotignore if it doesn't exist
        if not (root / DOT_IGNORE_FILENAME).is_file():
            with open(root / DOT_IGNORE_FILENAME, "w") as f:
                f.write("venv/\n")
                if (root / "bin" / "activate").is_file() or (root / "pyvenv.cfg").is_file():
                    # the user is developing inside venv
                    f.write("bin/\ninclude/\nlib/\npyvenv.cfg\n")

        repo = LocalSourceCodeDir(path=root, ignore_functions=ignore_functions)
        self._check_uploaded_folder(root, repo)
        requirements_file = root / "requirements.txt"
        # The entry point file needs to be relative to the root of the uploaded source file directory,
        # because the backend will invoke the lightning commands relative said source directory
        app_entrypoint_file = Path(self.entrypoint_file).absolute().relative_to(root)

        if name:
            # Override the name if provided by the CLI
            app_config.name = name

        print(f"The name of the app is: {app_config.name}")

        v1_env_vars = [V1EnvVar(name=k, value=v) for k, v in self.env_vars.items()]

        if len(self.secrets.values()) > 0:
            secret_names_to_ids = _names_to_ids(self.secrets.values())
            env_vars_from_secrets = [
                V1EnvVar(name=k, from_secret=secret_names_to_ids[v]) for k, v in self.secrets.items()
            ]
            v1_env_vars.extend(env_vars_from_secrets)

        if self.run_app_comment_commands or ENABLE_APP_COMMENT_COMMAND_EXECUTION:
            v1_env_vars.append(V1EnvVar(name="ENABLE_APP_COMMENT_COMMAND_EXECUTION", value="1"))

        if enable_multiple_works_in_default_container():
            v1_env_vars.append(V1EnvVar(name="ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", value="1"))

        if ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER:
            v1_env_vars.append(V1EnvVar(name="ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER", value="1"))

        if not ENABLE_PULLING_STATE_ENDPOINT:
            v1_env_vars.append(V1EnvVar(name="ENABLE_PULLING_STATE_ENDPOINT", value="0"))

        if not ENABLE_PUSHING_STATE_ENDPOINT:
            v1_env_vars.append(V1EnvVar(name="ENABLE_PUSHING_STATE_ENDPOINT", value="0"))

        works: List[V1Work] = _get_work_specs(self.app)

        # We need to collect a spec for each flow that contains a frontend so that the backend knows
        # for which flows it needs to start servers by invoking the cli (see the serve_frontend() method below)
        frontend_specs: List[V1Flowserver] = []
        for flow_name in self.app.frontends.keys():
            frontend_spec = V1Flowserver(name=flow_name)
            frontend_specs.append(frontend_spec)

        app_spec = V1LightningappInstanceSpec(
            app_entrypoint_file=str(app_entrypoint_file),
            enable_app_server=self.start_server,
            flow_servers=frontend_specs,
            desired_state=V1LightningappInstanceState.RUNNING,
            env=v1_env_vars,
            user_requested_flow_compute_config=V1UserRequestedFlowComputeConfig(
                name=self.app.flow_cloud_compute.name,
                shm_size=self.app.flow_cloud_compute.shm_size,
                preemptible=False,
            ),
        )

        # if requirements file at the root of the repository is present,
        # we pass just the file name to the backend, so backend can find it in the relative path
        if requirements_file.is_file():
            app_spec.image_spec = Gridv1ImageSpec(
                dependency_file_info=V1DependencyFileInfo(package_manager=V1PackageManager.PIP, path="requirements.txt")
            )
            if not DISABLE_DEPENDENCY_CACHE and not kwargs.get("no_cache"):
                # hash used for caching the dependencies
                app_spec.dependency_cache_key = get_hash(requirements_file)
        # we'll get the default project (quite similar to Github Organization) from the backend
        project = _get_project(self.backend.client)

        try:
            if cluster_id is not None:
                # Verify that the cluster exists
                list_clusters_resp = self.backend.client.cluster_service_list_clusters()
                cluster_ids = [cluster.id for cluster in list_clusters_resp.clusters]
                if cluster_id not in cluster_ids:
                    raise ValueError(f"You requested to run on cluster {cluster_id}, but that cluster doesn't exist.")

                self._ensure_cluster_project_binding(project.project_id, cluster_id)

            # Resolve the app name, instance, and cluster ID
            existing_instance = None
            app_name = app_config.name

            # List existing instances
            # TODO: Add pagination, otherwise this could break if users have a lot of apps.
            find_instances_resp = self.backend.client.lightningapp_instance_service_list_lightningapp_instances(
                project_id=project.project_id
            )

            # Seach for instances with the given name (possibly with some random characters appended)
            pattern = re.escape(f"{app_name}-") + ".{4}"
            instances = [
                lightningapp
                for lightningapp in find_instances_resp.lightningapps
                if lightningapp.name == app_name or (re.fullmatch(pattern, lightningapp.name) is not None)
            ]

            # If instances exist and cluster is None, mimic cluster selection logic to choose a default
            if cluster_id is None and len(instances) > 0:
                # Determine the cluster ID
                cluster_id = self._get_default_cluster(project.project_id)

            # If an instance exists on the cluster with the same base name - restart it
            for instance in instances:
                if instance.spec.cluster_id == cluster_id:
                    existing_instance = instance
                    break

            # If instances exist but not on the cluster - choose a randomised name
            if len(instances) > 0 and existing_instance is None:
                name_exists = True
                while name_exists:
                    random_name = self._randomise_name(app_name)
                    name_exists = any([instance.name == random_name for instance in instances])

                app_name = random_name

            # Create the app if it doesn't exist
            if existing_instance is None:
                app_body = Body7(name=app_name, can_download_source_code=True)
                lit_app = self.backend.client.lightningapp_v2_service_create_lightningapp_v2(
                    project_id=project.project_id, body=app_body
                )
                app_id = lit_app.id
            else:
                app_id = existing_instance.spec.app_id

            # check if user has sufficient credits to run an app
            # if so set the desired state to running otherwise, create the app in stopped state,
            # and open the admin ui to add credits and running the app.
            has_sufficient_credits = self._project_has_sufficient_credits(project, app=self.app)
            app_release_desired_state = (
                V1LightningappInstanceState.RUNNING if has_sufficient_credits else V1LightningappInstanceState.STOPPED
            )
            if not has_sufficient_credits:
                logger.warn("You may need Lightning credits to run your apps on the cloud.")

            # Stop the instance if it isn't stopped yet
            if existing_instance and existing_instance.status.phase != V1LightningappInstanceState.STOPPED:
                # TODO(yurij): Implement release switching in the UI and remove this
                # We can only switch release of the stopped instance
                existing_instance = self.backend.client.lightningapp_instance_service_update_lightningapp_instance(
                    project_id=project.project_id,
                    id=existing_instance.id,
                    body=Body3(spec=V1LightningappInstanceSpec(desired_state=V1LightningappInstanceState.STOPPED)),
                )
                # wait for the instance to stop for up to 150 seconds
                for _ in range(150):
                    existing_instance = self.backend.client.lightningapp_instance_service_get_lightningapp_instance(
                        project_id=project.project_id, id=existing_instance.id
                    )
                    if existing_instance.status.phase == V1LightningappInstanceState.STOPPED:
                        break
                    time.sleep(1)
                if existing_instance.status.phase != V1LightningappInstanceState.STOPPED:
                    raise RuntimeError("Failed to stop the existing instance.")

            network_configs: Optional[List[V1NetworkConfig]] = None
            if enable_multiple_works_in_default_container():
                network_configs = []
                initial_port = 8080 + 1 + len(frontend_specs)
                for _ in range(DEFAULT_NUMBER_OF_EXPOSED_PORTS):
                    network_configs.append(
                        V1NetworkConfig(
                            name="w" + str(initial_port),
                            port=initial_port,
                        )
                    )
                    initial_port += 1

            queue_server_type = V1QueueServerType.UNSPECIFIED
            if CLOUD_QUEUE_TYPE == "http":
                queue_server_type = V1QueueServerType.HTTP
            elif CLOUD_QUEUE_TYPE == "redis":
                queue_server_type = V1QueueServerType.REDIS

            release_body = Body8(
                app_entrypoint_file=app_spec.app_entrypoint_file,
                enable_app_server=app_spec.enable_app_server,
                flow_servers=app_spec.flow_servers,
                image_spec=app_spec.image_spec,
                cluster_id=cluster_id,
                network_config=network_configs,
                works=works,
                local_source=True,
                dependency_cache_key=app_spec.dependency_cache_key,
                user_requested_flow_compute_config=app_spec.user_requested_flow_compute_config,
                is_headless=_is_headless(self.app),
            )

            # create / upload the new app release
            lightning_app_release = self.backend.client.lightningapp_v2_service_create_lightningapp_release(
                project_id=project.project_id, app_id=app_id, body=release_body
            )

            if lightning_app_release.source_upload_url == "":
                raise RuntimeError("The source upload url is empty.")

            if getattr(lightning_app_release, "cluster_id", None):
                logger.info(f"Running app on {lightning_app_release.cluster_id}")

            # Save the config for re-runs
            app_config.save_to_dir(root)

            repo.package()
            repo.upload(url=lightning_app_release.source_upload_url)

            if existing_instance is not None:
                lightning_app_instance = (
                    self.backend.client.lightningapp_instance_service_update_lightningapp_instance_release(
                        project_id=project.project_id,
                        id=existing_instance.id,
                        body=Body4(release_id=lightning_app_release.id),
                    )
                )

                self.backend.client.lightningapp_instance_service_update_lightningapp_instance(
                    project_id=project.project_id,
                    id=existing_instance.id,
                    body=Body3(
                        spec=V1LightningappInstanceSpec(
                            desired_state=app_release_desired_state,
                            env=v1_env_vars,
                            queue_server_type=queue_server_type,
                        )
                    ),
                )
            else:
                lightning_app_instance = (
                    self.backend.client.lightningapp_v2_service_create_lightningapp_release_instance(
                        project_id=project.project_id,
                        app_id=app_id,
                        id=lightning_app_release.id,
                        body=Body9(
                            cluster_id=cluster_id,
                            desired_state=app_release_desired_state,
                            name=app_name,
                            env=v1_env_vars,
                            queue_server_type=queue_server_type,
                        ),
                    )
                )
        except ApiException as e:
            logger.error(e.body)
            sys.exit(1)

        if lightning_app_instance.status.phase == V1LightningappInstanceState.FAILED:
            raise RuntimeError("Failed to create the application. Cannot upload the source code.")

        if open_ui:
            click.launch(self._get_app_url(lightning_app_instance, not has_sufficient_credits))

        if cleanup_handle:
            cleanup_handle()

    def _ensure_cluster_project_binding(self, project_id: str, cluster_id: str):
        cluster_bindings = self.backend.client.projects_service_list_project_cluster_bindings(project_id=project_id)

        for cluster_binding in cluster_bindings.clusters:
            if cluster_binding.cluster_id != cluster_id:
                continue
            if cluster_binding.project_id == project_id:
                return

        self.backend.client.projects_service_create_project_cluster_binding(
            project_id=project_id,
            body=V1ProjectClusterBinding(cluster_id=cluster_id, project_id=project_id),
        )

    def _get_default_cluster(self, project_id: str) -> str:
        """This utility implements a minimal version of the cluster selection logic used in the cloud.

        TODO: This should be requested directly from the platform.
        """
        cluster_bindings = self.backend.client.projects_service_list_project_cluster_bindings(
            project_id=project_id
        ).clusters

        if not cluster_bindings:
            raise ValueError(f"No clusters are bound to the project {project_id}.")

        if len(cluster_bindings) == 1:
            return cluster_bindings[0].cluster_id

        clusters = [
            self.backend.client.cluster_service_get_cluster(cluster_binding.cluster_id)
            for cluster_binding in cluster_bindings
        ]

        # Filter global clusters
        clusters = [cluster for cluster in clusters if cluster.spec.cluster_type == V1ClusterType.GLOBAL]

        return random.choice(clusters).id

    @staticmethod
    def _randomise_name(app_name: str) -> str:
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return app_name + "-" + "".join(random.sample(letters, 4))

    @staticmethod
    def _check_uploaded_folder(root: Path, repo: LocalSourceCodeDir) -> None:
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

    def _project_has_sufficient_credits(self, project: V1Membership, app: Optional[LightningApp] = None):
        """check if user has enough credits to run the app with its hardware if app is not passed return True if
        user has 1 or more credits."""
        balance = project.balance
        if balance is None:
            balance = 0  # value is missing in some tests

        return balance >= 1

    @classmethod
    def load_app_from_file(cls, filepath: str) -> "LightningApp":
        """Load a LightningApp from a file, mocking the imports."""
        try:
            app = load_app_from_file(filepath, raise_exception=True, mock_imports=True)
        except FileNotFoundError as e:
            raise e
        except Exception:
            from lightning_app.testing.helpers import EmptyFlow

            # Create a generic app.
            logger.info("Could not load the app locally. Starting the app directly on the cloud.")
            app = LightningApp(EmptyFlow())
        return app

    @staticmethod
    def _get_app_url(lightning_app_instance: Externalv1LightningappInstance, need_credits: bool = False) -> str:
        action = "?action=add_credits" if need_credits else ""
        return f"{get_lightning_cloud_url()}/me/apps/{lightning_app_instance.id}{action}"


def _create_mount_drive_spec(work_name: str, mount: Mount) -> V1LightningworkDrives:
    if mount.protocol == "s3://":
        drive_type = V1DriveType.INDEXED_S3
        source_type = V1SourceType.S3
    else:
        raise RuntimeError(
            f"unknown mount protocol `{mount.protocol}`. Please verify this "
            f"drive type has been configured for use in the cloud dispatcher."
        )

    return V1LightningworkDrives(
        drive=V1Drive(
            metadata=V1Metadata(
                name=work_name,
            ),
            spec=V1DriveSpec(
                drive_type=drive_type,
                source_type=source_type,
                source=mount.source,
            ),
            status=V1DriveStatus(),
        ),
        mount_location=str(mount.mount_path),
    )


def _validate_build_spec_and_compute(work: LightningWork) -> None:
    if work.cloud_build_config.image is not None and work.cloud_compute.name == "default":
        raise ValueError(
            f"You requested a custom base image for the Work with name '{work.name}', but custom images are currently"
            " not supported on the default cloud compute instance. Please choose a different configuration, for example"
            " `CloudCompute('cpu-medium')`."
        )
