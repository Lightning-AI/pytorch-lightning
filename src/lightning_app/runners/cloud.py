import fnmatch
import logging
import os
import random
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from lightning_cloud.openapi import (
    Body3,
    Body4,
    Body7,
    Body8,
    Body9,
    Gridv1ImageSpec,
    V1BuildSpec,
    V1DependencyFileInfo,
    V1EnvVar,
    V1Flowserver,
    V1LightningappInstanceSpec,
    V1LightningappInstanceState,
    V1LightningworkSpec,
    V1NetworkConfig,
    V1PackageManager,
    V1PythonDependencyInfo,
    V1UserRequestedComputeConfig,
    V1Work,
)
from lightning_cloud.openapi.rest import ApiException

from lightning_app.core.constants import CLOUD_UPLOAD_WARNING, DISABLE_DEPENDENCY_CACHE
from lightning_app.runners.backends.cloud import CloudBackend
from lightning_app.runners.runtime import Runtime
from lightning_app.source_code import LocalSourceCodeDir
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.dependency_caching import get_hash
from lightning_app.utilities.packaging.app_config import AppConfig, find_config_file
from lightning_app.utilities.packaging.lightning_utils import _prepare_lightning_wheels_and_requirements

logger = logging.getLogger(__name__)


@dataclass
class CloudRuntime(Runtime):

    backend: Union[str, CloudBackend] = "cloud"

    def dispatch(
        self,
        on_before_run: Optional[Callable] = None,
        name: str = "",
        **kwargs: Any,
    ):
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
        config_file = find_config_file(self.entrypoint_file)
        app_config = AppConfig.load_from_file(config_file) if config_file else AppConfig()
        root = config_file.parent if config_file else Path(self.entrypoint_file).absolute().parent
        cleanup_handle = _prepare_lightning_wheels_and_requirements(root)
        repo = LocalSourceCodeDir(path=root)
        self._check_uploaded_folder(root, repo)
        requirements_file = root / "requirements.txt"
        # The entry point file needs to be relative to the root of the uploaded source file directory,
        # because the backend will invoke the lightning commands relative said source directory
        app_entrypoint_file = Path(self.entrypoint_file).absolute().relative_to(root)

        if name:
            # Override the name if provided by the CLI
            app_config.name = name

        app_config.save_to_dir(root)

        print(f"The name of the app is: {app_config.name}")

        work_reqs: List[V1Work] = []
        v1_env_vars = [V1EnvVar(name=k, value=v) for k, v in self.env_vars.items()]
        for flow in self.app.flows:
            for work in flow.works(recurse=False):
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
                random_name = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
                spec = V1LightningworkSpec(
                    build_spec=build_spec,
                    user_requested_compute_config=user_compute_config,
                    network_config=[V1NetworkConfig(name=random_name, port=work.port)],
                )
                work_reqs.append(V1Work(name=work.name, spec=spec))

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
                project.project_id, name=app_config.name
            )
            if list_apps_resp.lightningapps:
                # There can be only one app with unique project_id<>name pair
                lightning_app = list_apps_resp.lightningapps[0]
            else:
                app_body = Body7(name=app_config.name)
                lightning_app = self.backend.client.lightningapp_v2_service_create_lightningapp_v2(
                    project.project_id, app_body
                )

            release_body = Body8(
                app_entrypoint_file=app_spec.app_entrypoint_file,
                enable_app_server=app_spec.enable_app_server,
                flow_servers=app_spec.flow_servers,
                image_spec=app_spec.image_spec,
                works=[V1Work(name=work_req.name, spec=work_req.spec) for work_req in work_reqs],
                local_source=True,
                dependency_cache_key=app_spec.dependency_cache_key,
            )
            lightning_app_release = self.backend.client.lightningapp_v2_service_create_lightningapp_release(
                project.project_id, lightning_app.id, release_body
            )

            if lightning_app_release.source_upload_url == "":
                raise RuntimeError("The source upload url is empty.")

            repo.package()
            repo.upload(url=lightning_app_release.source_upload_url)

            # right now we only allow a single instance of the app
            find_instances_resp = self.backend.client.lightningapp_instance_service_list_lightningapp_instances(
                project.project_id, app_id=lightning_app.id
            )
            if find_instances_resp.lightningapps:
                existing_instance = find_instances_resp.lightningapps[0]
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
                            desired_state=V1LightningappInstanceState.RUNNING, env=v1_env_vars
                        )
                    ),
                )
            else:
                lightning_app_instance = (
                    self.backend.client.lightningapp_v2_service_create_lightningapp_release_instance(
                        project.project_id,
                        lightning_app.id,
                        lightning_app_release.id,
                        Body9(
                            desired_state=V1LightningappInstanceState.RUNNING, name=lightning_app.name, env=v1_env_vars
                        ),
                    )
                )
        except ApiException as e:
            logger.error(e.body)
            sys.exit(1)

        if on_before_run:
            on_before_run(lightning_app_instance)

        if lightning_app_instance.status.phase == V1LightningappInstanceState.FAILED:
            raise RuntimeError("Failed to create the application. Cannot upload the source code.")

        if cleanup_handle:
            cleanup_handle()

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
            logger.warning(warning_msg)
