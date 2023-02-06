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
from lightning_cloud.openapi import IdCodeconfigBody, V1MountPath
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cli_helpers import _error_and_exit
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("cloud_space_name", required=True)
@click.argument("source", required=True)
@click.argument("destination", required=True)
def mount(cloud_space_name: str, source: str, destination: str) -> List[str]:
    """Mount data to a CloudSpace."""

    if sys.platform == "win32":
        print("`ls` isn't supported on windows. Open an issue on Github.")
        sys.exit(0)

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:

        client = LightningClient()
        projects = client.projects_service_list_memberships()

        cloud_space = None

        for project in projects.memberships:

            project_id = project.project_id
            for cloud_space in client.cloud_space_service_list_cloud_spaces(project_id=project_id).cloudspaces:
                if cloud_space.name == cloud_space_name:
                    break

        if cloud_space is None:
            _error_and_exit(f"No CloudSpace with the name {cloud_space_name} was found.")

        mount_paths = cloud_space.code_config.mount_paths

        new_mount_path = V1MountPath(source=source.replace("s3://", ":s3:/"), destination=destination)

        if new_mount_path not in mount_paths:
            mount_paths.append(new_mount_path)

        if cloud_space.code_config.compute_config.name == "":
            cloud_space.code_config.compute_config.name = "cpu"

        live.stop()

        response = client.cloud_space_service_update_cloud_space_instance_config(
            project_id=project_id,
            id=cloud_space.id,
            body=IdCodeconfigBody(
                mount_paths=mount_paths,
                compute_config=cloud_space.code_config.compute_config,
            ),
        )

        print(response)
