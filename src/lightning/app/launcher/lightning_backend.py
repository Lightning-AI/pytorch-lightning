import inspect
import json
import logging
import os
import random
import string
import urllib
from time import monotonic, sleep, time
from typing import List, Optional

from lightning_cloud.openapi import (
    AppinstancesIdBody,
    Externalv1LightningappInstance,
    Externalv1Lightningwork,
    V1BuildSpec,
    V1Drive,
    V1DriveSpec,
    V1DriveStatus,
    V1DriveType,
    V1Flowserver,
    V1LightningappInstanceState,
    V1LightningappRestartPolicy,
    V1LightningworkClusterDriver,
    V1LightningworkDrives,
    V1LightningworkSpec,
    V1LightningworkState,
    V1ListLightningworkResponse,
    V1Metadata,
    V1NetworkConfig,
    V1PackageManager,
    V1PythonDependencyInfo,
    V1SourceType,
    V1UserRequestedComputeConfig,
)
from lightning_cloud.openapi.rest import ApiException

from lightning.app.core import LightningApp, LightningWork
from lightning.app.core.queues import QueuingSystem
from lightning.app.runners.backends.backend import Backend
from lightning.app.storage import Drive, Mount
from lightning.app.utilities.enum import WorkStageStatus, WorkStopReasons, make_status
from lightning.app.utilities.exceptions import LightningPlatformException
from lightning.app.utilities.network import LightningClient, _check_service_url_is_ready

logger = logging.getLogger(__name__)

from lightning_cloud.openapi import SpecLightningappInstanceIdWorksBody, WorksIdBody  # noqa: E402

LIGHTNING_STOP_TIMEOUT = int(os.getenv("LIGHTNING_STOP_TIMEOUT", 2 * 60))


def cloud_work_stage_to_work_status_stage(stage: V1LightningworkState) -> str:
    """Maps the Work stage names from the cloud backend to the status names in the Lightning framework."""
    mapping = {
        V1LightningworkState.STOPPED: WorkStageStatus.STOPPED,
        V1LightningworkState.PENDING: WorkStageStatus.PENDING,
        V1LightningworkState.NOT_STARTED: WorkStageStatus.PENDING,
        V1LightningworkState.IMAGE_BUILDING: WorkStageStatus.PENDING,
        V1LightningworkState.RUNNING: WorkStageStatus.RUNNING,
        V1LightningworkState.FAILED: WorkStageStatus.FAILED,
    }
    if stage not in mapping:
        raise ValueError(f"Cannot map the lightning-cloud work state {stage} to the lightning status stage.")
    return mapping[stage]


class CloudBackend(Backend):
    def __init__(
        self,
        entrypoint_file,
        queue_id: Optional[str] = None,
        status_update_interval: int = 5,
    ) -> None:
        # TODO: Properly handle queue_id in the cloud.
        super().__init__(entrypoint_file, queues=QueuingSystem("http"), queue_id=queue_id)
        self._status_update_interval = status_update_interval
        self._last_time_updated = None
        self.client = LightningClient(retry=True)
        self.base_url: Optional[str] = None

    @staticmethod
    def _work_to_spec(work: LightningWork) -> V1LightningworkSpec:
        work_requirements = "\n".join(work.cloud_build_config.requirements)

        build_spec = V1BuildSpec(
            commands=work.cloud_build_config.build_commands(),
            python_dependencies=V1PythonDependencyInfo(
                package_manager=V1PackageManager.PIP, packages=work_requirements
            ),
            image=work.cloud_build_config.image,
        )

        drive_specs: List[V1LightningworkDrives] = []
        for drive_attr_name, drive in [
            (k, getattr(work, k)) for k in work._state if isinstance(getattr(work, k), Drive)
        ]:
            if drive.protocol == "lit://":
                drive_type = V1DriveType.NO_MOUNT_S3
                source_type = V1SourceType.S3
            else:
                drive_type = V1DriveType.UNSPECIFIED
                source_type = V1SourceType.UNSPECIFIED

            drive_specs.append(
                V1LightningworkDrives(
                    drive=V1Drive(
                        metadata=V1Metadata(name=f"{work.name}.{drive_attr_name}"),
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

        # this should really be part of the work.cloud_compute struct, but to save
        # time we are not going to modify the backend in this set of PRs & instead
        # use the same s3 drives API which we used before.
        if work.cloud_compute.mounts is not None:
            if isinstance(work.cloud_compute.mounts, Mount):
                drive_specs.append(
                    _create_mount_drive_spec(
                        work_name=work.name,
                        mount=work.cloud_compute.mounts,
                    )
                )
            else:
                for mount in work.cloud_compute.mounts:
                    drive_specs.append(
                        _create_mount_drive_spec(
                            work_name=work.name,
                            mount=mount,
                        )
                    )

        if hasattr(work.cloud_compute, "interruptible"):
            preemptible = work.cloud_compute.interruptible
        else:
            preemptible = work.cloud_compute.preemptible

        colocation_group_id = None
        if hasattr(work.cloud_compute, "colocation_group_id"):
            colocation_group_id = work.cloud_compute.colocation_group_id

        user_compute_config = V1UserRequestedComputeConfig(
            name=work.cloud_compute.name,
            count=1,
            disk_size=work.cloud_compute.disk_size,
            preemptible=preemptible,
            shm_size=work.cloud_compute.shm_size,
            affinity_identifier=colocation_group_id,
        )

        random_name = "".join(random.choice(string.ascii_lowercase) for _ in range(5))  # noqa: S311

        return V1LightningworkSpec(
            build_spec=build_spec,
            drives=drive_specs,
            user_requested_compute_config=user_compute_config,
            network_config=[V1NetworkConfig(name=random_name, port=work.port)],
            desired_state=V1LightningworkState.RUNNING,
            restart_policy=V1LightningappRestartPolicy.NEVER,
            cluster_driver=V1LightningworkClusterDriver.DIRECT,
        )

    def create_work(self, app: LightningApp, work: LightningWork) -> None:
        app_id = self._get_app_id()
        project_id = self._get_project_id()
        list_response: V1ListLightningworkResponse = self.client.lightningwork_service_list_lightningwork(
            project_id=project_id, app_id=app_id
        )
        external_specs: List[Externalv1Lightningwork] = list_response.lightningworks

        # Find THIS work in the list of all registered works
        external_spec = None
        for es in external_specs:
            if es.name == work.name:
                external_spec = es
                break

        if external_spec is None:
            spec = self._work_to_spec(work)
            try:
                fn = SpecLightningappInstanceIdWorksBody.__init__
                params = list(inspect.signature(fn).parameters)
                extras = {}
                if "display_name" in params:
                    extras["display_name"] = getattr(work, "display_name", "")

                external_spec = self.client.lightningwork_service_create_lightningwork(
                    project_id=project_id,
                    spec_lightningapp_instance_id=app_id,
                    body=SpecLightningappInstanceIdWorksBody(
                        name=work.name,
                        spec=spec,
                        **extras,
                    ),
                )
                # overwriting spec with return value
                spec = external_spec.spec
            except ApiException as e:
                # We might get exceed quotas, or be out of credits.
                message = json.loads(e.body).get("message")
                raise LightningPlatformException(message) from None
        elif external_spec.spec.desired_state == V1LightningworkState.RUNNING:
            spec = external_spec.spec
            work._port = spec.network_config[0].port
        else:
            # Signal the LightningWorkState to go into state RUNNING
            spec = external_spec.spec

            # getting the updated spec but ignoring everything other than port & drives
            new_spec = self._work_to_spec(work)

            spec.desired_state = V1LightningworkState.RUNNING
            spec.network_config[0].port = new_spec.network_config[0].port
            spec.drives = new_spec.drives
            spec.user_requested_compute_config = new_spec.user_requested_compute_config
            spec.build_spec = new_spec.build_spec
            spec.env = new_spec.env
            try:
                self.client.lightningwork_service_update_lightningwork(
                    project_id=project_id,
                    id=external_spec.id,
                    spec_lightningapp_instance_id=app_id,
                    body=WorksIdBody(spec),
                )
            except ApiException as e:
                # We might get exceed quotas, or be out of credits.
                message = json.loads(e.body).get("message")
                raise LightningPlatformException(message) from None

        # Replace the undefined url and host by the known one.
        work._host = "0.0.0.0"  # noqa: S104
        work._future_url = f"{self._get_proxy_scheme()}://{spec.network_config[0].host}"

        # removing the backend to avoid the threadlock error
        _backend = work._backend
        work._backend = None
        app.work_queues[work.name].put(work)
        work._backend = _backend

        logger.info(f"Starting work {work.name}")
        logger.debug(f"With the following external spec: {external_spec}")

    def update_work_statuses(self, works: List[LightningWork]) -> None:
        """Pulls the status of each Work instance in the cloud.

        Normally, the Lightning frameworks communicates statuses through the queues, but while the Work instance is
        being provisionied, the queues don't exist yet and hence we need to make API calls directly to the backend to
        fetch the status and update it in the states.

        """
        if not works:
            return

        # TODO: should this run in a timer thread instead?
        if self._last_time_updated is not None and monotonic() - self._last_time_updated < self._status_update_interval:
            return

        cloud_work_specs = self._get_cloud_work_specs(self.client)
        local_works = works
        for cloud_work_spec in cloud_work_specs:
            for local_work in local_works:
                # TODO (tchaton) Better resolve pending status after succeeded

                # 1. Skip if the work isn't the current one.
                if local_work.name != cloud_work_spec.name:
                    continue

                # 2. Logic for idle timeout
                self._handle_idle_timeout(
                    local_work.cloud_compute.idle_timeout,
                    local_work,
                    cloud_work_spec,
                )

                # 3. Map the cloud phase to the local one
                cloud_stage = cloud_work_stage_to_work_status_stage(
                    cloud_work_spec.status.phase,
                )

                # 4. Detect if the work failed during pending phase
                if local_work.status.stage == WorkStageStatus.PENDING and cloud_stage in WorkStageStatus.FAILED:
                    if local_work._raise_exception:
                        raise Exception(f"The work {local_work.name} failed during pending phase.")
                    logger.error(f"The work {local_work.name} failed during pending phase.")

                # 5. Skip the pending and running as this is already handled by Lightning.
                if cloud_stage in (WorkStageStatus.PENDING, WorkStageStatus.RUNNING):
                    continue

                # TODO: Add the logic for wait_timeout
                if local_work.status.stage != cloud_stage:
                    latest_hash = local_work._calls["latest_call_hash"]
                    if latest_hash is None:
                        continue
                    local_work._calls[latest_hash]["statuses"].append(make_status(cloud_stage))

        self._last_time_updated = monotonic()

    def stop_all_works(self, works: List[LightningWork]) -> None:
        """Stop resources for all LightningWorks in this app.

        The Works are stopped rather than deleted so that they can be inspected for debugging.

        """
        cloud_works = self._get_cloud_work_specs(self.client)

        for cloud_work in cloud_works:
            self._stop_work(cloud_work)

        def all_works_stopped(works: List[Externalv1Lightningwork]) -> bool:
            for work in works:
                # deleted work won't be in the request hence only checking for stopped & failed
                if work.status.phase not in (
                    V1LightningworkState.STOPPED,
                    V1LightningworkState.FAILED,
                ):
                    return False
            return True

        t0 = time()
        while not all_works_stopped(self._get_cloud_work_specs(self.client)):
            # Wait a little..
            print("Waiting for works to stop...")
            sleep(3)

            # Break if we reached timeout.
            if time() - t0 > LIGHTNING_STOP_TIMEOUT:
                break

    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        if not self.base_url:
            self.base_url = base_url

        for flow in app.flows:
            if self.base_url:
                # Replacing the path with complete URL
                if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
                    raise ValueError(
                        "Base URL doesn't have a valid scheme, expected it to start with 'http://' or 'https://' "
                    )
                if isinstance(flow._layout, dict) and "target" not in flow._layout:
                    # FIXME: Why _check_service_url_is_ready doesn't work ?
                    frontend_url = urllib.parse.urljoin(self.base_url, flow.name + "/")
                    flow._layout["target"] = frontend_url

        for work in app.works:
            if (
                work._url == ""
                and work.status.stage
                in (
                    WorkStageStatus.RUNNING,
                    WorkStageStatus.SUCCEEDED,
                )
                and work._internal_ip != ""
                and _check_service_url_is_ready(f"http://{work._internal_ip}:{work._port}")
            ):
                work._url = work._future_url

    @staticmethod
    def _get_proxy_scheme() -> str:
        return os.environ.get("LIGHTNING_PROXY_SCHEME", "https")

    @staticmethod
    def _get_app_id() -> str:
        return os.environ["LIGHTNING_CLOUD_APP_ID"]

    @staticmethod
    def _get_project_id() -> str:
        return os.environ["LIGHTNING_CLOUD_PROJECT_ID"]

    @staticmethod
    def _get_cloud_work_specs(client: LightningClient) -> List[Externalv1Lightningwork]:
        list_response: V1ListLightningworkResponse = client.lightningwork_service_list_lightningwork(
            project_id=CloudBackend._get_project_id(),
            app_id=CloudBackend._get_app_id(),
        )
        return list_response.lightningworks

    def _handle_idle_timeout(self, idle_timeout: float, work: LightningWork, resp: Externalv1Lightningwork) -> None:
        if idle_timeout is None:
            return

        if work.status.stage != WorkStageStatus.SUCCEEDED:
            return

        if time() > (idle_timeout + work.status.timestamp):
            logger.info(f"Idle Timeout {idle_timeout} has triggered. Stopping gracefully the {work.name}.")
            latest_hash = work._calls["latest_call_hash"]
            status = make_status(WorkStageStatus.STOPPED, reason=WorkStopReasons.PENDING)
            work._calls[latest_hash]["statuses"].append(status)
            self._stop_work(resp)
            logger.debug(f"Stopping work: {resp.id}")

    def _register_queues(self, app, work):
        super()._register_queues(app, work)
        kw = {"queue_id": self.queue_id, "work_name": work.name}
        app.work_queues.update({work.name: self.queues.get_work_queue(**kw)})

    def stop_work(self, app: LightningApp, work: LightningWork) -> None:
        cloud_works = self._get_cloud_work_specs(self.client)
        for cloud_work in cloud_works:
            if work.name == cloud_work.name:
                self._stop_work(cloud_work)

    def _stop_work(self, work_resp: Externalv1Lightningwork) -> None:
        spec: V1LightningworkSpec = work_resp.spec
        if spec.desired_state == V1LightningworkState.DELETED:
            # work is set to be deleted. Do nothing
            return
        if spec.desired_state == V1LightningworkState.STOPPED:
            # work is set to be stopped already. Do nothing
            return
        if work_resp.status.phase == V1LightningworkState.FAILED:
            # work is already failed. Do nothing
            return
        spec.desired_state = V1LightningworkState.STOPPED
        self.client.lightningwork_service_update_lightningwork(
            project_id=CloudBackend._get_project_id(),
            id=work_resp.id,
            spec_lightningapp_instance_id=CloudBackend._get_app_id(),
            body=WorksIdBody(spec),
        )
        print(f"Stopping {work_resp.name} ...")

    def delete_work(self, app: LightningApp, work: LightningWork) -> None:
        cloud_works = self._get_cloud_work_specs(self.client)
        for cloud_work in cloud_works:
            if work.name == cloud_work.name:
                self._delete_work(cloud_work)

    def _delete_work(self, work_resp: Externalv1Lightningwork) -> None:
        spec: V1LightningworkSpec = work_resp.spec
        if spec.desired_state == V1LightningworkState.DELETED:
            # work is set to be deleted. Do nothing
            return
        spec.desired_state = V1LightningworkState.DELETED
        self.client.lightningwork_service_update_lightningwork(
            project_id=CloudBackend._get_project_id(),
            id=work_resp.id,
            spec_lightningapp_instance_id=CloudBackend._get_app_id(),
            body=WorksIdBody(spec),
        )
        print(f"Deleting {work_resp.name} ...")

    def update_lightning_app_frontend(self, app: "lightning.LightningApp"):  # noqa: F821
        """Used to create frontend's if the app couldn't be loaded locally."""
        if not len(app.frontends.keys()):
            return

        external_app_spec: "Externalv1LightningappInstance" = (
            self.client.lightningapp_instance_service_get_lightningapp_instance(
                project_id=CloudBackend._get_project_id(),
                id=CloudBackend._get_app_id(),
            )
        )

        frontend_specs = external_app_spec.spec.flow_servers
        spec = external_app_spec.spec
        if len(frontend_specs) != len(app.frontends.keys()):
            frontend_specs: List[V1Flowserver] = []
            for flow_name in sorted(app.frontends.keys()):
                frontend_spec = V1Flowserver(name=flow_name)
                frontend_specs.append(frontend_spec)

            spec.flow_servers = frontend_specs
            spec.enable_app_server = True

            logger.info("Found new frontends. Updating the app spec.")

            self.client.lightningapp_instance_service_update_lightningapp_instance(
                project_id=CloudBackend._get_project_id(),
                id=CloudBackend._get_app_id(),
                body=AppinstancesIdBody(spec=spec),
            )

    def stop_app(self, app: "lightning.LightningApp"):  # noqa: F821
        """Used to mark the App has stopped if everything has fine."""

        external_app_spec: "Externalv1LightningappInstance" = (
            self.client.lightningapp_instance_service_get_lightningapp_instance(
                project_id=CloudBackend._get_project_id(),
                id=CloudBackend._get_app_id(),
            )
        )

        spec = external_app_spec.spec
        spec.desired_state = V1LightningappInstanceState.STOPPED

        self.client.lightningapp_instance_service_update_lightningapp_instance(
            project_id=CloudBackend._get_project_id(),
            id=CloudBackend._get_app_id(),
            body=AppinstancesIdBody(spec=spec),
        )


def _create_mount_drive_spec(work_name: str, mount: "Mount") -> V1LightningworkDrives:
    if mount.protocol == "s3://":
        drive_type = V1DriveType.INDEXED_S3
        source_type = V1SourceType.S3
    else:
        raise RuntimeError(
            f"unknown mounts protocol `{mount.protocol}`. Please verify this "
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
