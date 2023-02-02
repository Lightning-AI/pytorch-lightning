import os

from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)


def pwd() -> None:

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True):

        cd_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")
        root = "."

        if not os.path.exists(cd_file):
            if path.startswith(".."):
                path = root

            with open(cd_file, "w") as f:
                f.write(path + "\n")
        else:
            with open(cd_file) as f:
                lines = f.readlines()
                root = lines[0].replace("\n", "")

    print(root)
