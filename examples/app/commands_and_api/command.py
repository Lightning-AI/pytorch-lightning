from argparse import ArgumentParser

from lightning.app.utilities.commands import ClientCommand
from pydantic import BaseModel


class CustomConfig(BaseModel):
    name: str


class CustomCommand(ClientCommand):
    description = "A command with a client."

    def run(self):
        parser = ArgumentParser()
        parser.add_argument("--name", type=str)
        args = parser.parse_args()
        self.invoke_handler(config=CustomConfig(name=args.name))
