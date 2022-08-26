from argparse import ArgumentParser

from pydantic import BaseModel

from lightning.app.utilities.commands import ClientCommand


class CustomConfig(BaseModel):
    name: str


class CustomCommand(ClientCommand):
    def run(self):
        parser = ArgumentParser()
        parser.add_argument("--name", type=str)
        args = parser.parse_args()
        self.invoke_handler(config=CustomConfig(name=args.name))
