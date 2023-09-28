from lightning.app import LightningApp, LightningFlow
from placeholdername import TemplateComponent


class LitApp(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.placeholdername = TemplateComponent()

    def run(self):
        print("this is a simple Lightning app to verify your component is working as expected")
        self.placeholdername.run()


app = LightningApp(LitApp())
