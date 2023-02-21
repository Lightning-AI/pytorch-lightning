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
import sys
from contextlib import nullcontext
from typing import Generator, List, Optional

import click
import lightning_cloud
import rich
from lightning_cloud.openapi import Externalv1LightningappInstance
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.connect.app import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cli_helpers import _error_and_exit
from lightning.app.utilities.network import LightningClient

_FOLDER_COLOR = "sky_blue1"
_FILE_COLOR = "white"

logger = Logger(__name__)


@click.argument("path", required=False)
def ls(path: Optional[str] = None, print: bool = True, use_live: bool = True) -> List[str]:
    """List the contents of a folder in the Lightning Cloud Filesystem."""

    from lightning.app.cli.commands.cd import _CD_FILE

    if sys.platform == "win32":
        _error_and_exit("`ls` isn't supported on windows. Open an issue on Github.")

    root = "/"

    context = (
        Live(Spinner("point", text=Text("pending...", style="white")), transient=True) if use_live else nullcontext()
    )

    with context:

        if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
            os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

        if not os.path.exists(_CD_FILE):
            with open(_CD_FILE, "w") as f:
                f.write(root + "\n")
        else:
            with open(_CD_FILE) as f:
                lines = f.readlines()
                root = lines[0].replace("\n", "")

        client = LightningClient(retry=False)
        projects = client.projects_service_list_memberships()

        if root == "/":
            project_names = [project.name for project in projects.memberships]
            if print:
                _print_names_with_colors(project_names, [_FOLDER_COLOR] * len(project_names))
            return project_names

        # Note: Root format has the following structure:
        # /{PROJECT_NAME}/{APP_NAME}/{ARTIFACTS_PATHS}
        splits = root.split("/")[1:]

        project = [project for project in projects.memberships if project.name == splits[0]]

        # This happens if the user changes cluster and the project doesn't exit.
        if len(project) == 0:
            return _error_and_exit(
                f"There isn't any Lightning Project matching the name {splits[0]}." " HINT: Use `lightning cd`."
            )

        project_id = project[0].project_id

        # Parallelise calls
        lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(
            project_id=project_id, async_req=True
        )
        lit_cloud_spaces = client.cloud_space_service_list_cloud_spaces(project_id=project_id, async_req=True)

        lit_apps = lit_apps.get().lightningapps
        lit_cloud_spaces = lit_cloud_spaces.get().cloudspaces

        if len(splits) == 1:
            apps = [lit_app.name for lit_app in lit_apps]
            cloud_spaces = [lit_cloud_space.name for lit_cloud_space in lit_cloud_spaces]
            ressource_names = sorted(set(cloud_spaces + apps))
            if print:
                _print_names_with_colors(ressource_names, [_FOLDER_COLOR] * len(ressource_names))
            return ressource_names

        lit_ressources = [lit_resource for lit_resource in lit_cloud_spaces if lit_resource.name == splits[1]]

        if len(lit_ressources) == 0:

            lit_ressources = [lit_resource for lit_resource in lit_apps if lit_resource.name == splits[1]]

            if len(lit_ressources) == 0:
                _error_and_exit(f"There isn't any Lightning Ressource matching the name {splits[1]}.")

        lit_resource = lit_ressources[0]

        app_paths = []
        app_colors = []

        cloud_spaces_paths = []
        cloud_spaces_colors = []

        depth = len(splits)

        prefix = "/".join(splits[2:])
        prefix = _get_prefix(prefix, lit_resource)

        for artifact in _collect_artifacts(client=client, project_id=project_id, prefix=prefix):

            if str(artifact.filename).startswith("/"):
                artifact.filename = artifact.filename[1:]

            path = os.path.join(project_id, prefix[1:], artifact.filename)

            artifact_splits = path.split("/")

            if len(artifact_splits) <= depth + 1:
                continue

            path = artifact_splits[depth + 1]

            paths = app_paths if isinstance(lit_resource, Externalv1LightningappInstance) else cloud_spaces_paths
            colors = app_colors if isinstance(lit_resource, Externalv1LightningappInstance) else cloud_spaces_colors

            if path not in paths:
                paths.append(path)

                # display files otherwise folders
                colors.append(_FILE_COLOR if len(artifact_splits) == depth + 1 else _FOLDER_COLOR)

    if print:
        if app_paths and cloud_spaces_paths:
            if app_paths:
                rich.print("Lightning App")
                _print_names_with_colors(app_paths, app_colors)

            if cloud_spaces_paths:
                rich.print("Lightning CloudSpaces")
                _print_names_with_colors(cloud_spaces_paths, cloud_spaces_colors)
        else:
            _print_names_with_colors(app_paths + cloud_spaces_paths, app_colors + cloud_spaces_colors)

    return app_paths + cloud_spaces_paths


def _add_colors(filename: str, color: Optional[str] = None) -> str:
    return f"[{color}]{filename}[/{color}]"


def _print_names_with_colors(names: List[str], colors: List[str], padding: int = 5) -> None:
    console = Console()
    width = console.width

    max_L = max([len(name) for name in names] + [0]) + padding

    use_spacing = False

    if max_L * len(names) < width:
        use_spacing = True

    num_cols = width // max_L

    columns = {}
    for index, (name, color) in enumerate(zip(names, colors)):
        row = index // num_cols
        if row not in columns:
            columns[row] = []
        columns[row].append((name, color))

    for row_index in sorted(columns):
        row = ""
        for (name, color) in columns[row_index]:
            if use_spacing:
                spacing = padding
            else:
                spacing = max_L - len(name)
            spaces = " " * spacing
            row += _add_colors(name, color) + spaces
        rich.print(row)


def _collect_artifacts(
    client: LightningClient,
    project_id: str,
    prefix: str = "",
    page_token: Optional[str] = "",
    cluster_id: Optional[str] = None,
    page_size: int = 100_000,
    tokens=None,
    include_download_url: bool = False,
) -> Generator:
    if tokens is None:
        tokens = []

    if cluster_id is None:
        clusters = client.projects_service_list_project_cluster_bindings(project_id)
        for cluster in clusters.clusters:
            yield from _collect_artifacts(
                client,
                project_id,
                prefix=prefix,
                cluster_id=cluster.cluster_id,
                page_token=page_token,
                tokens=tokens,
                page_size=page_size,
                include_download_url=include_download_url,
            )
    else:

        if page_token in tokens:
            return

        try:
            response = client.lightningapp_instance_service_list_project_artifacts(
                project_id,
                prefix=prefix,
                cluster_id=cluster_id,
                page_token=page_token,
                include_download_url=include_download_url,
                page_size=str(page_size),
            )
            for artifact in response.artifacts:
                if ".lightning-app-sync" in artifact.filename:
                    continue
                yield artifact

            if response.next_page_token != "":
                tokens.append(page_token)
                yield from _collect_artifacts(
                    client,
                    project_id,
                    prefix=prefix,
                    cluster_id=cluster_id,
                    page_token=response.next_page_token,
                    tokens=tokens,
                )
        except lightning_cloud.openapi.rest.ApiException:
            # Note: This is triggered when the request is wrong.
            # This is currently happening due to looping through the user clusters.
            pass


def _add_resource_prefix(prefix: str, resource_path: str):
    if resource_path in prefix:
        return prefix
    prefix = os.path.join(resource_path, prefix)
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix


def _get_prefix(prefix: str, lit_resource) -> str:
    if isinstance(lit_resource, Externalv1LightningappInstance):
        return _add_resource_prefix(prefix, f"lightningapps/{lit_resource.id}")

    return _add_resource_prefix(prefix, f"cloudspaces/{lit_resource.id}")
