import fnmatch
import os
import random
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import click
from lightning_cloud.openapi import (
    Body3,
    Body4,
    Body7,
    Body8,
    Body9,
    Gridv1ImageSpec,
    V1BuildSpec,
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
    ENABLE_APP_COMMENT_COMMAND_EXECUTION,
    ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER,
    ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER,
    ENABLE_PULLING_STATE_ENDPOINT,
    ENABLE_PUSHING_STATE_ENDPOINT,
)
from lightning_app.runners.backends.cloud import CloudBackend
from lightning_app.runners.runtime import Runtime
from lightning_app.source_code import LocalSourceCodeDir
from lightning_app.storage import Drive, Mount
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.dependency_caching import get_hash
from lightning_app.utilities.load_app import load_app_from_file
from lightning_app.utilities.packaging.app_config import _get_config_file, AppConfig
from lightning_app.utilities.packaging.lightning_utils import _prepare_lightning_wheels_and_requirements
from lightning_app.utilities.secrets import _names_to_ids

logger = Logger(__name__)


@dataclass
class CloudRuntime(Runtime):

    backend: Union[str, CloudBackend] = "cloud"

    def dispatch(
        self,
        on_before_run: Optional[Callable] = None,
        name: str = "",
        cluster_id: str = None,
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

        # TODO: verify lightning version
        # _verify_lightning_version()
        config_file = _get_config_file(self.entrypoint_file)
        app_config = AppConfig.load_from_file(config_file) if config_file.exists() else AppConfig()
        root = Path(self.entrypoint_file).absolute().parent
        cleanup_handle = _prepare_lightning_wheels_and_requirements(root)
        self.app._update_index_file()
        repo = LocalSourceCodeDir(path=root)
        self._check_uploaded_folder(root, repo)
        requirements_file = root / "requirements.txt"
        # The entry point file needs to be relative to the root of the uploaded source file directory,
        # because the backend will invoke the lightning commands relative said source directory
        app_entrypoint_file = Path(self.entrypoint_file).absolute().relative_to(root)

        if name:
            # Override the name if provided by the CLI
            app_config.name = name

        if cluster_id:
            # Override the cluster ID if provided by the CLI
            app_config.cluster_id = cluster_id

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

        if ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER:
            v1_env_vars.append(V1EnvVar(name="ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", value="1"))

        if ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER:
            v1_env_vars.append(V1EnvVar(name="ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER", value="1"))

        if not ENABLE_PULLING_STATE_ENDPOINT:
            v1_env_vars.append(V1EnvVar(name="ENABLE_PULLING_STATE_ENDPOINT", value="0"))

        if not ENABLE_PUSHING_STATE_ENDPOINT:
            v1_env_vars.append(V1EnvVar(name="ENABLE_PUSHING_STATE_ENDPOINT", value="0"))

        works: List[V1Work] = []
        for work in self.app.works:
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
                        mount_location=str(drive.root_folder),
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
            list_apps_resp = self.backend.client.lightningapp_v2_service_list_lightningapps_v2(
                project_id=project.project_id, name=app_config.name
            )
            if list_apps_resp.lightningapps:
                # There can be only one app with unique project_id<>name pair
                lit_app = list_apps_resp.lightningapps[0]
            else:
                app_body = Body7(name=app_config.name, can_download_source_code=True)
                lit_app = self.backend.client.lightningapp_v2_service_create_lightningapp_v2(
                    project_id=project.project_id, body=app_body
                )

            network_configs: Optional[List[V1NetworkConfig]] = None
            if ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER:
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

            # check if user has sufficient credits to run an app
            # if so set the desired state to running otherwise, create the app in stopped state,
            # and open the admin ui to add credits and running the app.
            has_sufficient_credits = self._project_has_sufficient_credits(project, app=self.app)
            app_release_desired_state = (
                V1LightningappInstanceState.RUNNING if has_sufficient_credits else V1LightningappInstanceState.STOPPED
            )
            if not has_sufficient_credits:
                logger.warn("You may need Lightning credits to run your apps on the cloud.")

            # right now we only allow a single instance of the app
            find_instances_resp = self.backend.client.lightningapp_instance_service_list_lightningapp_instances(
                project_id=project.project_id, app_id=lit_app.id
            )

            queue_server_type = V1QueueServerType.UNSPECIFIED
            if CLOUD_QUEUE_TYPE == "http":
                queue_server_type = V1QueueServerType.HTTP
            elif CLOUD_QUEUE_TYPE == "redis":
                queue_server_type = V1QueueServerType.REDIS

            if find_instances_resp.lightningapps:
                existing_instance = find_instances_resp.lightningapps[0]

                if not app_config.cluster_id:
                    # Re-run the app on the same cluster
                    app_config.cluster_id = existing_instance.spec.cluster_id

                if existing_instance.status.phase != V1LightningappInstanceState.STOPPED:
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

            if app_config.cluster_id is not None:
                # Verify that the cluster exists
                list_clusters_resp = self.backend.client.cluster_service_list_clusters()
                cluster_ids = [cluster.id for cluster in list_clusters_resp.clusters]
                if app_config.cluster_id not in cluster_ids:
                    if cluster_id:
                        msg = f"You requested to run on cluster {cluster_id}, but that cluster doesn't exist."
                    else:
                        msg = (
                            f"Your app last ran on cluster {app_config.cluster_id}, but that cluster "
                            "doesn't exist anymore."
                        )
                    click.confirm(
                        f"{msg} Do you want to run on Lightning Cloud instead?",
                        abort=True,
                        default=True,
                    )
                    app_config.cluster_id = None

            if app_config.cluster_id is not None:
                self._ensure_cluster_project_binding(project.project_id, app_config.cluster_id)

            release_body = Body8(
                app_entrypoint_file=app_spec.app_entrypoint_file,
                enable_app_server=app_spec.enable_app_server,
                flow_servers=app_spec.flow_servers,
                image_spec=app_spec.image_spec,
                cluster_id=app_config.cluster_id,
                network_config=network_configs,
                works=works,
                local_source=True,
                dependency_cache_key=app_spec.dependency_cache_key,
                user_requested_flow_compute_config=app_spec.user_requested_flow_compute_config,
            )

            # create / upload the new app release
            lightning_app_release = self.backend.client.lightningapp_v2_service_create_lightningapp_release(
                project_id=project.project_id, app_id=lit_app.id, body=release_body
            )

            if lightning_app_release.source_upload_url == "":
                raise RuntimeError("The source upload url is empty.")

            if getattr(lightning_app_release, "cluster_id", None):
                app_config.cluster_id = lightning_app_release.cluster_id
                logger.info(f"Running app on {lightning_app_release.cluster_id}")

            # Save the config for re-runs
            app_config.save_to_dir(root)

            repo.package()
            repo.upload(url=lightning_app_release.source_upload_url)

            if find_instances_resp.lightningapps:
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
                        app_id=lit_app.id,
                        id=lightning_app_release.id,
                        body=Body9(
                            cluster_id=app_config.cluster_id,
                            desired_state=app_release_desired_state,
                            name=lit_app.name,
                            env=v1_env_vars,
                            queue_server_type=queue_server_type,
                        ),
                    )
                )
        except ApiException as e:
            logger.error(e.body)
            sys.exit(1)

        if on_before_run:
            on_before_run(lightning_app_instance, need_credits=not has_sufficient_credits)

        if lightning_app_instance.status.phase == V1LightningappInstanceState.FAILED:
            raise RuntimeError("Failed to create the application. Cannot upload the source code.")

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

    @staticmethod
    def _check_uploaded_folder(root: Path, repo: LocalSourceCodeDir) -> None:
        """This method is used to inform the users if their folder files are large and how to filter them."""
        lightning_tar = set(fnmatch.filter(repo.files, "*lightning-*.tar.gz"))
        app_folder_size = sum(Path(p).stat().st_size for p in repo.files if p not in lightning_tar)
        app_folder_size_in_mb = round(app_folder_size / (1000 * 1000), 5)
        if app_folder_size_in_mb > CLOUD_UPLOAD_WARNING:
            path_sizes = [(p, Path(p).stat().st_size / (1000 * 1000)) for p in repo.files]
            largest_paths = sorted((x for x in path_sizes if x[-1] > 0.01), key=lambda x: x[1], reverse=True)[:25]
            largest_paths_msg = "\n".join(f"{round(s, 5)} MB: {p}" for p, s in largest_paths)
            warning_msg = (
                f"Your application folder {root} is more than {CLOUD_UPLOAD_WARNING} MB. "
                f"Found {app_folder_size_in_mb} MB \n"
                "Here are the largest files: \n"
                f"{largest_paths_msg}"
            )
            if not os.path.exists(os.path.join(root, ".lightningignore")):
                warning_msg = (
                    warning_msg
                    + "\nIn order to ignore some files or folder, "
                    + "create a `.lightningignore` file and add the paths to ignore."
                )
            else:
                warning_msg += "\nYou can ignore some files or folders by adding them to `.lightningignore`."

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
