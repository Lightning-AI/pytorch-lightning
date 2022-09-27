import os
import shutil
from typing import List, Optional, Tuple

import click

from lightning_app.utilities.cli_helpers import _retrieve_application_url_and_available_commands
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


@click.argument("app_name_or_id", required=True)
@click.option("-y", "--yes", required=False, is_flag=True, help="Whether to download the commands automatically.")
def connect(app_name_or_id: str, yes: bool = False):
    """Connect to a Lightning App."""
    from lightning_app.utilities.commands.base import _download_command

    home = os.path.expanduser("~")
    lightning_folder = os.path.join(home, ".lightning", "lightning_connection")

    if not os.path.exists(lightning_folder):
        os.makedirs(lightning_folder)

    connected_file = os.path.join(lightning_folder, "connect.txt")

    if os.path.exists(connected_file):
        with open(connected_file) as f:
            result = f.readlines()[0].replace("\n", "")

        if result == app_name_or_id:
            if app_name_or_id == "localhost":
                click.echo("You are connected to the local Lightning App.")
            else:
                click.echo(f"You are already connected to the cloud Lightning App: {app_name_or_id}.")
        else:
            disconnect()
            connect(app_name_or_id, yes)

    elif app_name_or_id.startswith("localhost"):

        if app_name_or_id != "localhost":
            raise Exception("You need to pass localhost to connect to the local Lightning App.")

        _, api_commands, __cached__ = _retrieve_application_url_and_available_commands(None)

        if api_commands is None:
            raise Exception(f"The commands weren't found. Is your app {app_name_or_id} running ?")

        commands_folder = os.path.join(lightning_folder, "commands")
        if not os.path.exists(commands_folder):
            os.makedirs(commands_folder)

        for command_name, metadata in api_commands.items():
            if "cls_path" in metadata:
                target_file = os.path.join(commands_folder, f"{command_name.replace(' ','_')}.py")
                _download_command(
                    command_name,
                    metadata["cls_path"],
                    metadata["cls_name"],
                    None,
                    target_file=target_file,
                )
                click.echo(f"Storing `{command_name}` under {target_file}")
                click.echo(f"You can review all the downloaded commands under {commands_folder} folder.")
            else:
                with open(os.path.join(commands_folder, f"{command_name}.txt"), "w") as f:
                    f.write(command_name)

        with open(connected_file, "w") as f:
            f.write(app_name_or_id + "\n")

        click.echo("You are connected to the local Lightning App.")
    else:
        _, api_commands, lightningapp_id = _retrieve_application_url_and_available_commands(app_name_or_id)

        if not api_commands:
            client = LightningClient()
            project = _get_project(client)
            lightningapps = client.lightningapp_instance_service_list_lightningapp_instances(
                project_id=project.project_id
            )
            click.echo(
                "We didn't find a matching App. Here are the available Apps that could be "
                f"connected to {[app.name for app in lightningapps.lightningapps]}."
            )
            return

        assert lightningapp_id

        if not yes:
            yes = click.confirm(
                f"The Lightning App `{app_name_or_id}` provides a command-line (CLI). "
                "Do you want to proceed and install its CLI ?"
            )
            click.echo(" ")

        if yes:
            commands_folder = os.path.join(lightning_folder, "commands")
            if not os.path.exists(commands_folder):
                os.makedirs(commands_folder)

            for command_name, metadata in api_commands.items():
                if "cls_path" in metadata:
                    target_file = os.path.join(commands_folder, f"{command_name}.py")
                    _download_command(
                        command_name,
                        metadata["cls_path"],
                        metadata["cls_name"],
                        lightningapp_id,
                        target_file=target_file,
                    )
                    click.echo(f"Storing `{command_name}` under {target_file}")
                    click.echo(f"You can review all the downloaded commands under {commands_folder} folder.")
                else:
                    with open(os.path.join(commands_folder, f"{command_name}.txt"), "w") as f:
                        f.write(command_name)

            click.echo(" ")
            click.echo("The client interface has been successfully installed. ")
            click.echo("You can now run the following commands:")
            for command in api_commands:
                click.echo(f"    lightning {command}")

        with open(connected_file, "w") as f:
            f.write(app_name_or_id + "\n")
            f.write(lightningapp_id + "\n")
        click.echo(" ")
        click.echo(f"You are connected to the cloud Lightning App: {app_name_or_id}.")


def disconnect(logout: bool = False):
    """Disconnect from an App."""
    home = os.path.expanduser("~")
    lightning_folder = os.path.join(home, ".lightning", "lightning_connection")
    connected_file = os.path.join(lightning_folder, "connect.txt")
    if os.path.exists(connected_file):
        with open(connected_file) as f:
            result = f.readlines()[0].replace("\n", "")

        os.remove(connected_file)
        commands_folder = os.path.join(lightning_folder, "commands")
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


def _retrieve_connection_to_an_app() -> Tuple[Optional[str], Optional[str]]:
    home = os.path.expanduser("~")
    lightning_folder = os.path.join(home, ".lightning", "lightning_connection")
    connected_file = os.path.join(lightning_folder, "connect.txt")

    if os.path.exists(connected_file):
        with open(connected_file) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
            if len(lines) == 2:
                return lines[0], lines[1]
            return lines[0], None
    return None, None


def _get_commands_folder() -> str:
    home = os.path.expanduser("~")
    lightning_folder = os.path.join(home, ".lightning", "lightning_connection")
    return os.path.join(lightning_folder, "commands")


def _resolve_command_path(command: str) -> str:
    return os.path.join(_get_commands_folder(), f"{command}.py")


def _list_app_commands() -> List[str]:
    command_names = sorted(
        n.replace(".py", "").replace(".txt", "").replace("_", " ")
        for n in os.listdir(_get_commands_folder())
        if n != "__pycache__"
    )
    if not command_names:
        click.echo("The current Lightning App doesn't have commands.")
        return []

    click.echo("Usage: lightning [OPTIONS] COMMAND [ARGS]...")
    click.echo("")
    click.echo("  --help     Show this message and exit.")
    click.echo("")
    click.echo("Lightning App Commands")
    max_length = max(len(n) for n in command_names)
    for command_name in command_names:
        padding = (max_length + 1 - len(command_name)) * " "
        click.echo(f"  {command_name}{padding}Description")
    return command_names
