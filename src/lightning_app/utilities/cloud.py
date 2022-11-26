import os
import warnings

from lightning_cloud.openapi import V1Membership

import lightning_app
from lightning_app.core.constants import LIGHTNING_CLOUD_PROJECT_ID
from lightning_app.utilities.enum import AppStage
from lightning_app.utilities.network import LightningClient


def _get_project(client: LightningClient, project_id: str = LIGHTNING_CLOUD_PROJECT_ID) -> V1Membership:
    """Get a project membership for the user from the backend."""
    projects = client.projects_service_list_memberships()
    if project_id is not None:
        for membership in projects.memberships:
            if membership.project_id == project_id:
                break
        else:
            raise ValueError(
                "Environment variable `LIGHTNING_CLOUD_PROJECT_ID` is set but could not find an associated project."
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


def is_running_in_cloud() -> bool:
    """Returns True if the Lightning App is running in the cloud."""
    return "LIGHTNING_APP_STATE_URL" in os.environ
