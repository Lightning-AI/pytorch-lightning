# example_app.py

from pathlib import Path

import lightning as L
from lightning.app import frontend


class YourComponent(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.message_to_print = "Hello World!"
        self.should_print = False

    def configure_layout(self):
        return frontend.StaticWebFrontend(Path(__file__).parent / "ui/dist")


class HelloLitReact(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.react_ui = YourComponent()

    def run(self):
        if self.react_ui.should_print:
            print(f"{self.counter}: {self.react_ui.message_to_print}")
            self.counter += 1

    def configure_layout(self):
        return [{"name": "React UI", "content": self.react_ui}]


app = L.LightningApp(HelloLitReact())
