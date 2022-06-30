from placeholdername import ComponentA, ComponentB

import lightning_app as la


class LitApp(la.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component_a = ComponentA()
        self.component_b = ComponentB()

    def run(self):
        self.component_a.run()
        self.component_b.run()


app = la.LightningApp(LitApp())
