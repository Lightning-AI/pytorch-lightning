import os

from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.cd import _CD_FILE
from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)


def pwd() -> str:

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True):

        root = _pwd()

    print(root)

    return root


def _pwd() -> str:
    root = "/"

    if not os.path.exists(_CD_FILE):
        with open(_CD_FILE, "w") as f:
            f.write(root + "\n")
    else:
        with open(_CD_FILE) as f:
            lines = f.readlines()
            root = lines[0].replace("\n", "")

    return root
