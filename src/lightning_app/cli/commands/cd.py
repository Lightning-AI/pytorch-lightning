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
from typing import Optional, Tuple, Union

import click
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning_app.cli.commands import ls
from lightning_app.cli.connect.app import _LIGHTNING_CONNECTION_FOLDER
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cli_helpers import _error_and_exit

logger = Logger(__name__)

_HOME = os.path.expanduser("~")
_CD_FILE = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")


@click.argument("path", nargs=-1)
def cd(path: Optional[Union[Tuple[str], str]], verify: bool = True) -> None:
    """Change the current directory within the Lightning Cloud filesystem."""

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:

        root = "/"

        if isinstance(path, Tuple) and len(path) > 0:
            path = " ".join(path)

        # handle ~/
        if isinstance(path, str) and path.startswith(_HOME):
            path = "/" + path.replace(_HOME, "")

        # handle no path -> /
        if path is None or len(path) == 0:
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

            if verify:
                if path.startswith("/"):
                    paths = [os.path.join(path, p) for p in ls.ls(path, print=False, use_live=False)]
                else:
                    paths = [os.path.join(root, p) for p in ls.ls(root, print=False, use_live=False)]

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
                if path.startswith(".."):
                    root = _apply_double_dots(root, path)
                elif path.startswith("~"):
                    root = path[2:]
                else:
                    root = os.path.join(root, path)

            if verify and root != "/" and not any(p.startswith(root) or root.startswith(p) for p in paths):
                _error_and_exit(f"no such file or directory: {path}")

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
