from lightning.app import LightningApp, LightningFlow
from placeholdername import ComponentA, ComponentB


class LitApp(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component_a = ComponentA()
        self.component_b = ComponentB()

    def run(self):
        self.component_a.run()
        self.component_b.run()


app = LightningApp(LitApp())
