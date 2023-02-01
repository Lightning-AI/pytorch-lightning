import click
import os
from lightning.app.utilities.app_helpers import Logger
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER

logger = Logger(__name__)

def pwd() -> None:

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
    
    print(root)
