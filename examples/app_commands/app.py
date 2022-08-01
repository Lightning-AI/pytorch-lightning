from command import CustomCommand, CustomConfig

from lightning import LightningFlow
from lightning_app.core.app import LightningApp


class ChildFlow(LightningFlow):
    def trigger_method(self, name: str):
        print(f"Hello {name}")

    def configure_commands(self):
        return [{"nested_trigger_command": self.trigger_method}]


class FlowCommands(LightningFlow):
    def __init__(self):
        super().__init__()
        self.names = []
        self.child_flow = ChildFlow()

    def run(self):
        if len(self.names):
            print(self.names)

    def trigger_without_client_command(self, name: str):
        self.names.append(name)

    def trigger_with_client_command(self, config: CustomConfig):
        self.names.append(config.name)

    def configure_commands(self):
        commands = [
            {"trigger_without_client_command": self.trigger_without_client_command},
            {"trigger_with_client_command": CustomCommand(self.trigger_with_client_command)},
        ]
        return commands + self.child_flow.configure_commands()


app = LightningApp(FlowCommands())
