import json
from datetime import datetime
from typing import List

from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    Externalv1Lightningwork,
    V1ClusterType,
    V1GetClusterResponse,
    V1LightningappInstanceState,
    V1LightningappInstanceStatus,
)
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lightning_app.cli.cmd_clusters import AWSClusterManager
from lightning_app.cli.core import Formatable
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


class _AppManager:
    """_AppManager implements API calls specific to Lightning AI BYOC apps."""

    def __init__(self) -> None:
        self.api_client = LightningClient()

    def get_cluster(self, cluster_id: str) -> V1GetClusterResponse:
        return self.api_client.cluster_service_get_cluster(id=cluster_id)

    def get_app(self, app_id: str) -> Externalv1LightningappInstance:
        project = _get_project(self.api_client)
        return self.api_client.lightningapp_instance_service_get_lightningapp_instance(
            project_id=project.project_id, id=app_id
        )

    def list_apps(
        self, cluster_id: str = None, limit: int = 100, phase_in: List[str] = []
    ) -> List[Externalv1LightningappInstance]:
        project = _get_project(self.api_client)

        kwargs = {
            "project_id": project.project_id,
            "limit": limit,
            "phase_in": phase_in,
        }
        if cluster_id is not None:
            kwargs["cluster_id"] = cluster_id

        resp = self.api_client.lightningapp_instance_service_list_lightningapp_instances(**kwargs)
        apps = resp.lightningapps
        while resp.next_page_token is not None and resp.next_page_token != "":
            kwargs["page_token"] = resp.next_page_token
            resp = self.api_client.lightningapp_instance_service_list_lightningapp_instances(**kwargs)
            apps = apps + resp.lightningapps
        return apps

    def list_components(self, app_id: str) -> List[Externalv1Lightningwork]:
        project = _get_project(self.api_client)
        resp = self.api_client.lightningwork_service_list_lightningwork(project_id=project.project_id, app_id=app_id)
        return resp.lightningworks

    def list(self, cluster_id: str = None, limit: int = 100) -> None:
        console = Console()
        console.print(_AppList(self.list_apps(cluster_id=cluster_id, limit=limit)).as_table())

    def delete(self, cluster_id: str, app_id: str) -> None:
        console = Console()

        cl = AWSClusterManager()
        default_cluster = None
        valid_clusters = []
        for cluster in cl.get_clusters().clusters:
            valid_clusters.append(cluster.id)
            if cluster.spec.cluster_type == V1ClusterType.GLOBAL and default_cluster is None:
                default_cluster = cluster.id

        if cluster_id is None:
            cluster_id = default_cluster

        if cluster_id not in valid_clusters:
            err = ValueError(
                f"Could not delete app {app_id} because the cluster {cluster_id} does not exist. "
                f"Please re-run the `lightning delete app` command specifying one of the available "
                f"clusters: {valid_clusters}"
            )
            print(err)
            return

        apps = self.list_apps(cluster_id=cluster_id)
        valid_app_names = [app.name for app in apps]
        valid_app_ids = [app.id for app in apps]
        if (app_id not in valid_app_ids) and (app_id not in valid_app_names):
            err = ValueError(
                f"Could not delete app {app_id} because there is no app by that name or ID on the "
                f"{cluster_id} cluster. Please run `lightning list apps` to view a list of valid apps which "
                f"can be deleted on this cluster."
            )
            print(err)
            return

        console.print(f"App: {app_id} has been successfully deleted from cluster: {cluster_id}")
        return


class _AppList(Formatable):
    def __init__(self, apps: List[Externalv1LightningappInstance]) -> None:
        self.apps = apps

    @staticmethod
    def _textualize_state_transitions(
        desired_state: V1LightningappInstanceState, current_state: V1LightningappInstanceStatus
    ) -> Text:
        phases = {
            V1LightningappInstanceState.IMAGE_BUILDING: Text("building image", style="bold yellow"),
            V1LightningappInstanceState.PENDING: Text("pending", style="bold yellow"),
            V1LightningappInstanceState.RUNNING: Text("running", style="bold green"),
            V1LightningappInstanceState.FAILED: Text("failed", style="bold red"),
            V1LightningappInstanceState.STOPPED: Text("stopped"),
            V1LightningappInstanceState.NOT_STARTED: Text("not started"),
            V1LightningappInstanceState.DELETED: Text("deleted", style="bold red"),
            V1LightningappInstanceState.UNSPECIFIED: Text("unspecified", style="bold red"),
        }

        if current_state.phase == V1LightningappInstanceState.UNSPECIFIED and current_state.start_timestamp is None:
            return Text("not yet started", style="bold yellow")

        if (
            desired_state == V1LightningappInstanceState.DELETED
            and current_state.phase != V1LightningappInstanceState.DELETED
        ):
            return Text("terminating", style="bold red")

        if (
            any(
                phase == current_state.phase
                for phase in [V1LightningappInstanceState.PENDING, V1LightningappInstanceState.STOPPED]
            )
            and desired_state == V1LightningappInstanceState.RUNNING
        ):
            return Text("restarting", style="bold yellow")

        return phases[current_state.phase]

    def as_json(self) -> str:
        return json.dumps(self.apps)

    def as_table(self) -> Table:
        table = Table("id", "name", "status", "cluster", "created", show_header=True, header_style="bold green")

        for app in self.apps:
            status = self._textualize_state_transitions(desired_state=app.spec.desired_state, current_state=app.status)

            # this guard is necessary only until 0.3.93 releases which includes the `created_at`
            # field to the external API
            created_at = datetime.now()
            if hasattr(app, "created_at"):
                created_at = app.created_at

            table.add_row(
                app.id,
                app.name,
                status,
                app.spec.cluster_id,
                created_at.strftime("%Y-%m-%d") if created_at else "",
            )
        return table
