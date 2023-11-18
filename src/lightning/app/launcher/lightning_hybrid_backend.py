import os
from typing import Optional

from lightning_cloud.openapi import AppinstancesIdBody, Externalv1LightningappInstance

from lightning.app.core import constants
from lightning.app.core.queues import QueuingSystem
from lightning.app.launcher.lightning_backend import CloudBackend
from lightning.app.runners.backends.backend import Backend
from lightning.app.runners.backends.mp_process import MultiProcessingBackend
from lightning.app.utilities.network import LightningClient

if hasattr(constants, "get_cloud_queue_type"):
    CLOUD_QUEUE_TYPE = constants.get_cloud_queue_type() or "redis"
else:
    CLOUD_QUEUE_TYPE = "redis"


class CloudHybridBackend(Backend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, queues=QueuingSystem(CLOUD_QUEUE_TYPE), **kwargs)
        cloud_backend = CloudBackend(*args, **kwargs)
        kwargs.pop("queue_id")
        multiprocess_backend = MultiProcessingBackend(*args, **kwargs)

        self.backends = {"cloud": cloud_backend, "multiprocess": multiprocess_backend}
        self.work_to_network_configs = {}

    def create_work(self, app, work) -> None:
        backend = self._get_backend(work)
        if isinstance(backend, MultiProcessingBackend):
            self._prepare_work_creation(app, work)
        backend.create_work(app, work)

    def _prepare_work_creation(self, app, work) -> None:
        app_id = self._get_app_id()
        project_id = self._get_project_id()
        assert project_id

        client = LightningClient()
        list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
        lit_app: Optional[Externalv1LightningappInstance] = None

        for lapp in list_apps_resp.lightningapps:
            if lapp.id == app_id:
                lit_app = lapp

        assert lit_app

        network_configs = lit_app.spec.network_config

        index = len(self.work_to_network_configs)

        if work.name not in self.work_to_network_configs:
            self.work_to_network_configs[work.name] = network_configs[index]

        # Enable Ingress and update the specs.
        lit_app.spec.network_config[index].enable = True

        client.lightningapp_instance_service_update_lightningapp_instance(
            project_id=project_id,
            id=lit_app.id,
            body=AppinstancesIdBody(name=lit_app.name, spec=lit_app.spec),
        )

        work_network_config = self.work_to_network_configs[work.name]

        work._host = "0.0.0.0"  # noqa: S104
        work._port = work_network_config.port
        work._future_url = f"{self._get_proxy_scheme()}://{work_network_config.host}"

    def update_work_statuses(self, works) -> None:
        if works:
            backend = self._get_backend(works[0])
            backend.update_work_statuses(works)

    def stop_all_works(self, works) -> None:
        if works:
            backend = self._get_backend(works[0])
            backend.stop_all_works(works)

    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        works = app.works
        if works:
            backend = self._get_backend(works[0])
            backend.resolve_url(app, base_url)

    def update_lightning_app_frontend(self, app: "lightning.LightningApp"):  # noqa: F821
        self.backends["cloud"].update_lightning_app_frontend(app)

    def stop_work(self, app, work) -> None:
        backend = self._get_backend(work)
        if isinstance(backend, MultiProcessingBackend):
            self._prepare_work_stop(app, work)
        backend.stop_work(app, work)

    def delete_work(self, app, work) -> None:
        backend = self._get_backend(work)
        if isinstance(backend, MultiProcessingBackend):
            self._prepare_work_stop(app, work)
        backend.delete_work(app, work)

    def _prepare_work_stop(self, app, work):
        app_id = self._get_app_id()
        project_id = self._get_project_id()
        assert project_id

        client = LightningClient()
        list_apps_resp = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
        lit_app: Optional[Externalv1LightningappInstance] = None

        for lapp in list_apps_resp.lightningapps:
            if lapp.id == app_id:
                lit_app = lapp

        assert lit_app

        network_config = self.work_to_network_configs[work.name]

        for nc in lit_app.spec.network_config:
            if nc.host == network_config.host:
                nc.enable = False

        client.lightningapp_instance_service_update_lightningapp_instance(
            project_id=project_id,
            id=lit_app.id,
            body=AppinstancesIdBody(name=lit_app.name, spec=lit_app.spec),
        )

        del self.work_to_network_configs[work.name]

    def _register_queues(self, app, work):
        backend = self._get_backend(work)
        backend._register_queues(app, work)

    def _get_backend(self, work):
        if work.cloud_compute.id == "default":
            return self.backends["multiprocess"]
        return self.backends["cloud"]

    @staticmethod
    def _get_proxy_scheme() -> str:
        return os.environ.get("LIGHTNING_PROXY_SCHEME", "https")

    @staticmethod
    def _get_app_id() -> str:
        return os.environ["LIGHTNING_CLOUD_APP_ID"]

    @staticmethod
    def _get_project_id() -> str:
        return os.environ["LIGHTNING_CLOUD_PROJECT_ID"]

    def stop_app(self, app: "lightning.LightningApp"):  # noqa: F821
        """Used to mark the App has stopped if everything has fine."""
        self.backends["cloud"].stop_app(app)
