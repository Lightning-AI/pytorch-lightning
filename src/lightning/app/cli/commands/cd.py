import click
import os
from typing import Optional
from lightning.app.utilities.app_helpers import Logger
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER

logger = Logger(__name__)

_HOME = os.path.expanduser("~")

@click.argument("path", required=False)
def cd(path: Optional[str] = None) -> None:

    cd_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")

    if isinstance(path, str) and path.startswith(_HOME):
        path =  "/" + path.replace(_HOME, '')

    if path is None:
        path = "/"

    if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
        os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

    if not os.path.exists(cd_file):
        if path.startswith(".."):
            path = root

        with open(cd_file, "w") as f:
            f.write(path + "\n")

        print(f"cd {path}")
    else:
        with open(cd_file, "r") as f:
            lines = f.readlines()
            root = lines[0].replace("\n", "")

        if root == "/":
            if path == '/':
                root = "/"
            elif not path.startswith(".."):
                if not path.startswith("/"):
                    path = "/" + path
                root = path
        else:
            # TODO: Validate the new path exists
            if path.startswith(".."):
                root = "/".join(root.split('/')[:-1])
            elif path.startswith("~"):
                root = path[2:]
            else:
                root = os.path.join(root, path)

        os.remove(cd_file)

        with open(cd_file, "w") as f:
            f.write(root + "\n")

        print(f"cd {root}")