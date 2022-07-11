from lightning import LightningFlow
from lightning_app.core.app import LightningApp


class ChildFlow(LightningFlow):
    def trigger_method(self, name: str):
        print(name)

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

    def trigger_method(self, name: str):
        self.names.append(name)

    def configure_commands(self):
        return [{"flow_trigger_command": self.trigger_method}] + self.child_flow.configure_commands()


app = LightningApp(FlowCommands())
