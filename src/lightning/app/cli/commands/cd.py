import os
from typing import Optional

import click
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)

_HOME = os.path.expanduser("~")
_CD_FILE = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")


@click.argument("path", required=False)
def cd(path: Optional[str] = None) -> str:
    """
    Command to nagivate through the Lightning Cloud filesystem.
    """

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:

        root = "/"

        # handle ~/
        if isinstance(path, str) and path.startswith(_HOME):
            path = "/" + path.replace(_HOME, "")

        # handle no path -> /
        if path is None:
            path = "/"

        if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
            os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

        if not os.path.exists(_CD_FILE):
            # Start from the root
            if path.startswith(".."):
                root = _apply_double_dots(root, path)

            with open(_CD_FILE, "w") as f:
                f.write(root + "\n")

            live.stop()

            print(f"cd {root}")

            return root
        else:
            # read from saved cd
            with open(_CD_FILE) as f:
                lines = f.readlines()
                root = lines[0].replace("\n", "")

            # generate new root
            if root == "/":
                if path == "/":
                    root = "/"
                elif not path.startswith(".."):
                    if not path.startswith("/"):
                        path = "/" + path
                    root = path
                else:
                    root = _apply_double_dots(root, path)
            else:
                # TODO: Validate the new path exists
                if path.startswith(".."):
                    root = _apply_double_dots(root, path)
                elif path.startswith("~"):
                    root = path[2:]
                else:
                    root = os.path.join(root, path)

            os.remove(_CD_FILE)

            # store new root
            with open(_CD_FILE, "w") as f:
                f.write(root + "\n")

        live.stop()

        print(f"cd {root}")

    return root


def _apply_double_dots(root: str, path: str) -> str:
    splits = [split for split in path.split("/") if split != ""]
    for split in splits:
        if split == "..":
            root = "/" + os.path.join(*root.split("/")[:-1])
        else:
            root = os.path.join(root, split)
    return root
