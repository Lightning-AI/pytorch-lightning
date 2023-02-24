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

import os
from typing import Optional

from lightning_cloud.openapi import V1Membership

import lightning.app
from lightning.app.core.constants import LIGHTNING_CLOUD_PROJECT_ID
from lightning.app.utilities.enum import AppStage
from lightning.app.utilities.network import LightningClient


def _get_project(client: LightningClient, project_id: Optional[str] = None, verbose: bool = True) -> V1Membership:
    """Get a project membership for the user from the backend."""
    if project_id is None:
        project_id = LIGHTNING_CLOUD_PROJECT_ID

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
        if verbose:
            print(f"Defaulting to the project: {projects.memberships[0].name}")
    return projects.memberships[0]


def _sigterm_flow_handler(*_, app: "lightning.app.LightningApp"):
    app.stage = AppStage.STOPPING


def is_running_in_cloud() -> bool:
    """Returns True if the Lightning App is running in the cloud."""
    return bool(int(os.environ.get("LAI_RUNNING_IN_CLOUD", "0"))) or "LIGHTNING_APP_STATE_URL" in os.environ
