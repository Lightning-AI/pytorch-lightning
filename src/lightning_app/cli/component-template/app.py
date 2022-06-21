from placeholdername import TemplateComponent

import lightning_app as la


class LitApp(la.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.placeholdername = TemplateComponent()

    def run(self):
        print("this is a simple Lightning app to verify your component is working as expected")
        self.placeholdername.run()


app = la.LightningApp(LitApp())
