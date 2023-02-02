from argparse import ArgumentParser
from uuid import uuid4

from pydantic import BaseModel

from lightning.app.utilities.commands import ClientCommand


class RunNotebookConfig(BaseModel):
    name: str
    cloud_compute: str


class RunNotebook(ClientCommand):
    description = "Run a Notebook."

    def run(self):
        # 1. Define your own argument parser. You can use argparse, click, etc...
        parser = ArgumentParser(description='Run Notebook Parser')
        parser.add_argument("--name", type=str, default=None)
        parser.add_argument("--cloud_compute", type=str, default="cpu")
        hparams = parser.parse_args()

        # 2. Invoke the server side handler by sending a payload.
        response = self.invoke_handler(
            config=RunNotebookConfig(
                name=hparams.name or str(uuid4()),
                cloud_compute=hparams.cloud_compute,
            ),
        )

        # 3. Print the server response.
        print(response)
