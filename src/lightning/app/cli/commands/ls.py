import os
from typing import List, Optional

import click
import rich
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.cd import _CD_FILE
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.network import LightningClient

_FOLDER_COLOR = "blue"
_FILE_COLOR = "white"

logger = Logger(__name__)


@click.argument("path", required=False)
def ls(path: Optional[str] = None) -> List[str]:
    """List the contents of a folder in the Lightning Cloud Filesystem."""

    root = "/"
    paths = []

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
            project_names = [_add_colors(project.name, color=_FOLDER_COLOR) for project in projects.memberships]
            rich.print(*sorted(set(project_names)))
            return project_names

        # Note: Root format has the following structure:
        # /{PROJECT_NAME}/{APP_NAME}/{ARTIFACTS_PATHS}
        # TODO: Add support for CloudSpaces, etc..
        splits = root.split("/")[1:]

        project_id = [project.project_id for project in projects.memberships if project.name == splits[0]][0]

        lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps

        if len(splits) == 1:
            app_names = sorted([_add_colors(lit_app.name, color=_FOLDER_COLOR) for lit_app in lit_apps])
            rich.print(*app_names)
            return app_names

        lit_apps = [lit_app for lit_app in lit_apps if lit_app.name == splits[1]]
        assert len(lit_apps) == 1
        lit_app = lit_apps[0]

        depth = len(splits)
        subpath = "/".join(splits[2:])
        # TODO: Replace with project level endpoints
        response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id, lit_app.id)
        for artifact in response.artifacts:
            path = os.path.join(project_id, lit_app.name, artifact.filename)
            artifact_splits = path.split("/")

            if len(artifact_splits) < depth + 1:
                continue

            if not str(artifact.filename).startswith(subpath):
                continue

            # display files otherwise folders
            if len(artifact_splits) == depth + 1:
                color = _FILE_COLOR
            else:
                color = _FOLDER_COLOR

            paths.append(_add_colors(artifact_splits[depth], color=color))

        paths = sorted(set(paths))

    rich.print(*paths)

    return paths


def _add_colors(filename: str, color: Optional[str] = None) -> str:
    return f"[{color}]{filename}[/{color}]"
