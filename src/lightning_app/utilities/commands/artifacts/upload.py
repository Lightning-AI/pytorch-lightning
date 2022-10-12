import argparse
import os
import re
from typing import List

from pydantic import BaseModel

from lightning.app.storage.path import filesystem, shared_storage_path
from lightning.app.utilities.commands import ClientCommand


class UploadArtifactsCommand(ClientCommand):
    def run(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--files", nargs="+", default=[], help="Provide a list of files.")
        hparams = parser.parse_args()


SHOW_ARTIFACT = {"show artifacts": UploadArtifactsCommand(None)}
