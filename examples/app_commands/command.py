from pydantic import BaseModel

from lightning_app.utilities.commands import ClientCommand


class CustomConfig(BaseModel):
    name: str


class CustomCommand(ClientCommand):
    def run(self, name: str):
        self.invoke_handler(config=CustomConfig(name=name))
