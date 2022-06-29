import os

import lightning as L
from lightning.app.frontend import StaticWebFrontend


class NestedFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()

    def configure_layout(self):
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), "build"))


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.my_flow = NestedFlow()

    def configure_layout(self):
        return {"name": "Docs", "content": self.my_flow}


app = L.LightningApp(Flow())
