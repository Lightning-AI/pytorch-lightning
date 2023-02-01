import click
from typing import Optional
from lightning.app.utilities.cloud import _get_project
import os
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER

logger = Logger(__name__)


@click.argument("path", required=True)
def cd(path: str) -> None:

    cd_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")
    root = '.'

    if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
        os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

    if not os.path.exists(cd_file):
        with open(cd_file, "w") as f:
            f.write(root)
    else:
        with open(cd_file, "r") as f:
            lines = f.readlines()
            root = lines[0].replace("\n", "")

        os.remove(cd_file)

        with open(cd_file, "w") as f:
            f.write(root)