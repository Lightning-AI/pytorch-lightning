import json
from datetime import datetime

from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1LightningappInstanceState,
    V1LightningappInstanceStatus,
)
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lightning_app.cli.core import Formatable
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


class _AppManager:
    """_AppManager implements API calls specific to Lightning AI BYOC apps."""

    def __init__(self):
        self.api_client = LightningClient()

    def list(self, cluster_id: str = None, limit: int = 100):
        project = _get_project(self.api_client)

        args = {
            "project_id": project.project_id,
            "limit": limit,
        }
        if cluster_id is not None:
            args["cluster_id"] = cluster_id

        resp = self.api_client.lightningapp_instance_service_list_lightningapp_instances(**args)
        apps = resp.lightningapps
        while resp.next_page_token is not None and resp.next_page_token != "":
            args["page_token"] = resp.next_page_token
            resp = self.api_client.lightningapp_instance_service_list_lightningapp_instances(**args)
            apps = apps + resp.lightningapps
        console = Console()
        console.print(_AppList(resp.lightningapps).as_table())


class _AppList(Formatable):
    def __init__(self, apps: [Externalv1LightningappInstance]):
        self.apps = apps

    @staticmethod
    def _textualize_state_transitions(
        desired_state: V1LightningappInstanceState, current_state: V1LightningappInstanceStatus
    ):
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
            app: Externalv1LightningappInstance
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
