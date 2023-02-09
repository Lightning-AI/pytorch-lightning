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

import json
import os
import shutil
import sys
from subprocess import Popen
from typing import List, Optional, Tuple

import click
import psutil
from lightning_utilities.core.imports import package_available
from rich.progress import Progress

from lightning.app.utilities.cli_helpers import _get_app_display_name, _LightningAppOpenAPIRetriever
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.enum import OpenAPITags
from lightning.app.utilities.log import get_logfile
from lightning.app.utilities.network import LightningClient

_HOME = os.path.expanduser("~")
_PPID = os.getenv("LIGHTNING_CONNECT_PPID", str(psutil.Process(os.getpid()).ppid()))
_LIGHTNING_CONNECTION = os.path.join(_HOME, ".lightning", "lightning_connection")
_LIGHTNING_CONNECTION_FOLDER = os.path.join(_LIGHTNING_CONNECTION, _PPID)


@click.argument("app_name_or_id", required=True)
def connect_app(app_name_or_id: str):
    """Connect your local terminal to a running lightning app.

    After connecting, the lightning CLI will respond to commands exposed by the app.

    Example:

    \b
    # connect to an app named pizza-cooker-123
    lightning connect pizza-cooker-123
    \b
    # this will now show the commands exposed by pizza-cooker-123
    lightning --help
    \b
    # while connected, you can run the cook-pizza command exposed
    # by pizza-cooker-123.BTW, this should arguably generate an exception :-)
    lightning cook-pizza --flavor pineapple
    \b
    # once done, disconnect and go back to the standard lightning CLI commands
    lightning disconnect
    """
    from lightning.app.utilities.commands.base import _download_command

    _clean_lightning_connection()

    if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
        os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

    connected_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "connect.txt")

    matched_connection_path = _scan_lightning_connections(app_name_or_id)

    if os.path.exists(connected_file):
        with open(connected_file) as f:
            result = f.readlines()[0].replace("\n", "")

        if result == app_name_or_id:
            if app_name_or_id == "localhost":
                click.echo("You are connected to the local Lightning App.")
            else:
                click.echo(f"You are already connected to the cloud Lightning App: {app_name_or_id}.")
        else:
            disconnect_app()
            connect_app(app_name_or_id)

    elif app_name_or_id.startswith("localhost"):

        with Progress() as progress_bar:
            connecting = progress_bar.add_task("[magenta]Setting things up for you...", total=1.0)

            if app_name_or_id != "localhost":
                raise Exception("You need to pass localhost to connect to the local Lightning App.")

            retriever = _LightningAppOpenAPIRetriever(None)

            if retriever.api_commands is None:
                raise Exception(f"Connection wasn't successful. Is your app {app_name_or_id} running?")

            increment = 1 / (1 + len(retriever.api_commands))

            progress_bar.update(connecting, advance=increment)

            commands_folder = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "commands")
            if not os.path.exists(commands_folder):
                os.makedirs(commands_folder)

            _write_commands_metadata(retriever.api_commands)

            with open(os.path.join(commands_folder, "openapi.json"), "w") as f:
                json.dump(retriever.openapi, f)

            _install_missing_requirements(retriever)

            for command_name, metadata in retriever.api_commands.items():
                if "cls_path" in metadata:
                    target_file = os.path.join(commands_folder, f"{command_name.replace(' ','_')}.py")
                    _download_command(
                        command_name,
                        metadata["cls_path"],
                        metadata["cls_name"],
                        None,
                        target_file=target_file,
                    )
                else:
                    with open(os.path.join(commands_folder, f"{command_name}.txt"), "w") as f:
                        f.write(command_name)

                progress_bar.update(connecting, advance=increment)

        with open(connected_file, "w") as f:
            f.write(app_name_or_id + "\n")

        click.echo("The lightning CLI now responds to app commands. Use 'lightning --help' to see them.")
        click.echo(" ")

        Popen(
            f"LIGHTNING_CONNECT_PPID={_PPID} {sys.executable} -m lightning --help",
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).wait()

    elif matched_connection_path:

        matched_connected_file = os.path.join(matched_connection_path, "connect.txt")
        matched_commands = os.path.join(matched_connection_path, "commands")
        if os.path.isdir(matched_commands):
            commands = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "commands")
            shutil.copytree(matched_commands, commands)
            shutil.copy(matched_connected_file, connected_file)

        click.echo("The lightning CLI now responds to app commands. Use 'lightning --help' to see them.")
        click.echo(" ")

        Popen(
            f"LIGHTNING_CONNECT_PPID={_PPID} {sys.executable} -m lightning --help",
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).wait()

    else:
        with Progress() as progress_bar:
            connecting = progress_bar.add_task("[magenta]Setting things up for you...", total=1.0)

            retriever = _LightningAppOpenAPIRetriever(app_name_or_id)

            if not retriever.api_commands:
                client = LightningClient(retry=False)
                project = _get_project(client)
                apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project.project_id)
                click.echo(
                    "We didn't find a matching App. Here are the available Apps that you can "
                    f"connect to {[_get_app_display_name(app) for app in apps.lightningapps]}."
                )
                return

            increment = 1 / (1 + len(retriever.api_commands))

            progress_bar.update(connecting, advance=increment)

            _install_missing_requirements(retriever)

            commands_folder = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "commands")
            if not os.path.exists(commands_folder):
                os.makedirs(commands_folder)

            _write_commands_metadata(retriever.api_commands)

            for command_name, metadata in retriever.api_commands.items():
                if "cls_path" in metadata:
                    target_file = os.path.join(commands_folder, f"{command_name}.py")
                    _download_command(
                        command_name,
                        metadata["cls_path"],
                        metadata["cls_name"],
                        retriever.app_id,
                        target_file=target_file,
                    )
                else:
                    with open(os.path.join(commands_folder, f"{command_name}.txt"), "w") as f:
                        f.write(command_name)

                progress_bar.update(connecting, advance=increment)

        with open(connected_file, "w") as f:
            f.write(retriever.app_name + "\n")
            f.write(retriever.app_id + "\n")

        click.echo("The lightning CLI now responds to app commands. Use 'lightning --help' to see them.")
        click.echo(" ")

        Popen(
            f"LIGHTNING_CONNECT_PPID={_PPID} {sys.executable} -m lightning --help",
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).wait()


def disconnect_app(logout: bool = False):
    """Disconnect from an App."""
    _clean_lightning_connection()

    connected_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "connect.txt")
    if os.path.exists(connected_file):
        with open(connected_file) as f:
            result = f.readlines()[0].replace("\n", "")

        os.remove(connected_file)
        commands_folder = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "commands")
        if os.path.exists(commands_folder):
            shutil.rmtree(commands_folder)

        if result == "localhost":
            click.echo("You are disconnected from the local Lightning App.")
        else:
            click.echo(f"You are disconnected from the cloud Lightning App: {result}.")
    else:
        if not logout:
            click.echo(
                "You aren't connected to any Lightning App. "
                "Please use `lightning connect app_name_or_id` to connect to one."
            )


def _read_connected_file(connected_file):
    if os.path.exists(connected_file):
        with open(connected_file) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
            if len(lines) == 2:
                return lines[0], lines[1]
            return lines[0], None
    return None, None


def _retrieve_connection_to_an_app() -> Tuple[Optional[str], Optional[str]]:
    connected_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "connect.txt")
    return _read_connected_file(connected_file)


def _get_commands_folder() -> str:
    return os.path.join(_LIGHTNING_CONNECTION_FOLDER, "commands")


def _write_commands_metadata(api_commands):
    metadata = {command_name: metadata for command_name, metadata in api_commands.items()}
    metadata_path = os.path.join(_get_commands_folder(), ".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def _get_commands_metadata():
    metadata_path = os.path.join(_get_commands_folder(), ".meta.json")
    with open(metadata_path) as f:
        return json.load(f)


def _resolve_command_path(command: str) -> str:
    return os.path.join(_get_commands_folder(), f"{command}.py")


def _list_app_commands(echo: bool = True) -> List[str]:
    metadata = _get_commands_metadata()
    metadata = {key.replace("_", " "): value for key, value in metadata.items()}

    command_names = list(sorted(metadata.keys()))
    if not command_names:
        click.echo("The current Lightning App doesn't have commands.")
        return []

    app_info = metadata[command_names[0]].get("app_info", None)

    title, description, on_connect_end = "Lightning", None, None
    if app_info:
        title = app_info.get("title")
        description = app_info.get("description")
        on_connect_end = app_info.get("on_connect_end")

    if echo:
        click.echo(f"{title} App")
        if description:
            click.echo("")
            click.echo("Description:")
            if description.endswith("\n"):
                description = description[:-2]
            click.echo(f"  {description}")
        click.echo("")
        click.echo("Commands:")
        max_length = max(len(n) for n in command_names)
        for command_name in command_names:
            padding = (max_length + 1 - len(command_name)) * " "
            click.echo(f"  {command_name}{padding}{metadata[command_name].get('description', '')}")
        if "LIGHTNING_CONNECT_PPID" in os.environ and on_connect_end:
            if on_connect_end.endswith("\n"):
                on_connect_end = on_connect_end[:-2]
            click.echo(on_connect_end)
    return command_names


def _install_missing_requirements(
    retriever: _LightningAppOpenAPIRetriever,
    fail_if_missing: bool = False,
):
    requirements = set()
    for metadata in retriever.api_commands.values():
        if metadata["tag"] == OpenAPITags.APP_CLIENT_COMMAND:
            for req in metadata.get("requirements", []) or []:
                requirements.add(req)

    if requirements:
        missing_requirements = []
        for req in requirements:
            if not (package_available(req) or package_available(req.replace("-", "_"))):
                missing_requirements.append(req)

        if missing_requirements:
            if fail_if_missing:
                missing_requirements = " ".join(missing_requirements)
                print(f"The command failed as you are missing the following requirements: `{missing_requirements}`.")
                sys.exit(0)

            for req in missing_requirements:
                std_out_out = get_logfile("output.log")
                with open(std_out_out, "wb") as stdout:
                    Popen(
                        f"{sys.executable} -m pip install {req}",
                        shell=True,
                        stdout=stdout,
                        stderr=stdout,
                    ).wait()
                os.remove(std_out_out)


def _clean_lightning_connection():
    if not os.path.exists(_LIGHTNING_CONNECTION):
        return

    for ppid in os.listdir(_LIGHTNING_CONNECTION):
        try:
            psutil.Process(int(ppid))
        except (psutil.NoSuchProcess, ValueError):
            connection = os.path.join(_LIGHTNING_CONNECTION, str(ppid))
            if os.path.exists(connection):
                shutil.rmtree(connection)


def _scan_lightning_connections(app_name_or_id):
    if not os.path.exists(_LIGHTNING_CONNECTION):
        return

    for ppid in os.listdir(_LIGHTNING_CONNECTION):
        try:
            psutil.Process(int(ppid))
        except (psutil.NoSuchProcess, ValueError):
            continue

        connection_path = os.path.join(_LIGHTNING_CONNECTION, str(ppid))

        connected_file = os.path.join(connection_path, "connect.txt")
        curr_app_name, curr_app_id = _read_connected_file(connected_file)

        if not curr_app_name:
            continue

        if app_name_or_id == curr_app_name or app_name_or_id == curr_app_id:
            return connection_path

    return None
