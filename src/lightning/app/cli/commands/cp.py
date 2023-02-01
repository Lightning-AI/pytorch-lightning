import click
import os
import sys
from pathlib import Path
from lightning.app.utilities.app_helpers import Logger
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)

@click.argument("local_path", required=True)
@click.argument("remote_path", required=True)
def cp(local_path: str, remote_path: str) -> None:

    cd_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")
    root = '.'
    
    if not os.path.exists(cd_file):
        if path.startswith(".."):
            path = root

        with open(cd_file, "w") as f:
            f.write(path + "\n")
    else:
        with open(cd_file, "r") as f:
            lines = f.readlines()
            root = lines[0].replace("\n", "")

    local_path = Path(local_path).resolve()

    if not local_path.exists():
        print(f"FileNotFoundError: The provided local path {local_path} doesn't exist.")
        sys.exit(0)

    if remote_path == ".":
        remote_path = root

    answer = None

    while answer not in ['yes', 'no']:
        answer = click.prompt(f"Do you want to copy files from {local_path} to {remote_path} ? Answer by yes/no")

    if answer == "no":
        print(f"Canceling copy.")
        sys.exit(0)

    client = LightningClient()
    if not project_id:
        project_id = _get_project(client).project_id

    lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps
