from command import CustomCommand, CustomConfig
from lightning import LightningFlow
from lightning.app.api import Get, Post
from lightning.app.core.app import LightningApp


async def handler():
    print("Has been called")
    return "Hello World !"


class ChildFlow(LightningFlow):
    def nested_command(self, name: str):
        """A nested command."""
        print(f"Hello {name}")

    def configure_commands(self):
        return [{"nested_command": self.nested_command}]


class FlowCommands(LightningFlow):
    def __init__(self):
        super().__init__()
        self.names = []
        self.child_flow = ChildFlow()

    def run(self):
        if self.names:
            print(self.names)

    def command_without_client(self, name: str):
        """A command without a client."""
        self.names.append(name)

    def command_with_client(self, config: CustomConfig):
        self.names.append(config.name)

    def configure_commands(self):
        commands = [
            {"command_without_client": self.command_without_client},
            {"command_with_client": CustomCommand(self.command_with_client)},
        ]
        return commands + self.child_flow.configure_commands()

    def configure_api(self):
        return [
            Post("/user/command_without_client", self.command_without_client),
            Get("/pure_function", handler),
        ]


app = LightningApp(FlowCommands(), log_level="debug")
