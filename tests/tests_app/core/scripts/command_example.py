from lightning import LightningFlow
from lightning_app.core.app import LightningApp


class FlowCommands(LightningFlow):
    def __init__(self):
        super().__init__()
        self.names = []

    def run(self):
        if len(self.names):
            print(self.names)
            self._exit()

    def trigger_method(self, name: str):
        self.names.append(name)

    def configure_commands(self):
        return [{"user_command": self.trigger_method}]


app = LightningApp(FlowCommands())
