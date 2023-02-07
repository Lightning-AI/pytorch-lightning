# Copyright The Lightning team.
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

import sys
from typing import List

import click
import rich
from lightning_cloud.openapi import ProjectIdDataConnectionsBody
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cli_helpers import _error_and_exit
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("name", required=True)
@click.argument("source", required=True)
@click.argument("destination", required=False)
@click.argument("project_name", required=False)
def connect_data(
    name: str,
    source: str,
    destination: str = "",
    project_name: str = "",
) -> List[str]:
    """Create a new data connection."""

    if sys.platform == "win32":
        print("`ls` isn't supported on windows. Open an issue on Github.")
        sys.exit(0)

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True):

        client = LightningClient()
        projects = client.projects_service_list_memberships()

        project_id = None

        for project in projects.memberships:

            if project.name == project_name:
                project_id = project.project_id

        if project_id is None:
            project_id = _get_project(client).project_id

        if not source.startswith("s3://"):
            _error_and_exit(
                "Only public s3 folder are supported for now. Please, open a Github issue with your use case."
            )

        try:
            client.data_connection_service_create_data_connection(
                project_id=project_id,
                body=ProjectIdDataConnectionsBody(
                    name=name,
                    source=source.replace("s3://", ":s3:/"),
                    destination=destination,
                ),
            )
        except Exception:
            _error_and_exit("The data connection creation failed.")

    rich.print("[green]Succeeded[/green]: You have created a new data connection.")
