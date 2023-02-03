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

import os
import sys
from typing import Generator, List, Optional

import click
import rich
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.cd import _CD_FILE
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.network import LightningClient

_FOLDER_COLOR = "sky_blue1"
_FILE_COLOR = "white"

logger = Logger(__name__)


@click.argument("path", required=False)
def ls(path: Optional[str] = None) -> List[str]:
    """List the contents of a folder in the Lightning Cloud Filesystem."""

    if sys.platform == "win32":
        print("`ls` isn't supported on windows. Open an issue on Github.")
        sys.exit(0)

    root = "/"

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:

        live.stop()

        if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
            os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

        if not os.path.exists(_CD_FILE):
            with open(_CD_FILE, "w") as f:
                f.write(root + "\n")
        else:
            with open(_CD_FILE) as f:
                lines = f.readlines()
                root = lines[0].replace("\n", "")

        client = LightningClient()
        projects = client.projects_service_list_memberships()

        if root == "/":
            project_names = [project.name for project in projects.memberships]
            _print_names_with_colors(project_names, [_FOLDER_COLOR] * len(project_names))
            return project_names

        # Note: Root format has the following structure:
        # /{PROJECT_NAME}/{APP_NAME}/{ARTIFACTS_PATHS}
        # TODO: Add support for CloudSpaces, etc..
        splits = root.split("/")[1:]

        project_id = [project.project_id for project in projects.memberships if project.name == splits[0]][0]

        lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps

        if len(splits) == 1:
            app_names = sorted([lit_app.name for lit_app in lit_apps])
            _print_names_with_colors(app_names, [_FOLDER_COLOR] * len(app_names))
            return app_names

        lit_apps = [lit_app for lit_app in lit_apps if lit_app.name == splits[1]]

        if len(lit_apps) != 1:
            print(f"ERROR: There isn't any Lightning App matching the name {splits[1]}.")
            sys.exit(0)

        lit_app = lit_apps[0]

        paths = []
        colors = []
        depth = len(splits)
        subpath = "/".join(splits[2:])
        # TODO: Replace with project level endpoints
        for artifact in _collect_artifacts(client, project_id, lit_app.id):
            path = os.path.join(project_id, lit_app.name, artifact.filename)
            artifact_splits = path.split("/")

            if len(artifact_splits) < depth + 1:
                continue

            if not str(artifact.filename).startswith(subpath):
                continue

            path = artifact_splits[depth]

            if path not in paths:
                paths.append(path)

                # display files otherwise folders
                colors.append(_FILE_COLOR if len(artifact_splits) == depth + 1 else _FOLDER_COLOR)

    _print_names_with_colors(paths, colors)

    return paths


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
    app_id: str,
    page_token: Optional[str] = "",
    tokens=None,
) -> Generator:
    if tokens is None:
        tokens = []

    if page_token in tokens:
        return

    response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(
        project_id, app_id, page_token=page_token
    )
    yield from response.artifacts

    if response.next_page_token != "":
        tokens.append(page_token)
        yield from _collect_artifacts(client, project_id, app_id, page_token=response.next_page_token, tokens=tokens)
