import warnings

from lightning_cloud.openapi.rest import ApiException
from lightning_cloud.openapi import V1Membership, Body3, V1LightningappInstanceSpec, V1LightningappInstanceState

import lightning_app
from lightning_app.core.constants import LIGHTNING_CLOUD_PROJECT_ID
from lightning_app.utilities.enum import AppStage
from lightning_app.utilities.network import LightningClient


def stop_app_cloud(app_id: str, project_id: str, client: LightningClient):
    """Sends an API request to the cloud backend to mark app with the given ID as stopped."""
    updated_spec = V1LightningappInstanceSpec(desired_state=V1LightningappInstanceState.STOPPED)

    try:
        client.lightningapp_instance_service_update_lightningapp_instance(project_id, app_id, Body3(spec=updated_spec))
    except ApiException as e:
        # FIXME(alecmerdler): Raise a different exception to hide API error from end user...
        raise e


def _get_project(client: LightningClient) -> V1Membership:
    """Get a project membership for the user from the backend."""
    projects = client.projects_service_list_memberships()
    if LIGHTNING_CLOUD_PROJECT_ID is not None:
        for membership in projects.memberships:
            if membership.project_id == LIGHTNING_CLOUD_PROJECT_ID:
                break
        else:
            raise ValueError(
                "Environment variable LIGHTNING_CLOUD_PROJECT_ID is set " "but could not find an associated project."
            )
        return membership

    if len(projects.memberships) == 0:
        raise ValueError("No valid projects found. Please reach out to lightning.ai team to create a project")
    if len(projects.memberships) > 1:
        warnings.warn(
            f"It is currently not supported to have multiple projects but "
            f"found {len(projects.memberships)} projects."
            f" Defaulting to the project {projects.memberships[0].name}"
        )
    return projects.memberships[0]


def _sigterm_flow_handler(*_, app: "lightning_app.LightningApp"):
    app.stage = AppStage.STOPPING
