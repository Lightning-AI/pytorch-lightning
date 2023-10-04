from lightning.app import LightningWork


class TemplateComponent(LightningWork):
    def __init__(self) -> None:
        super().__init__()
        self.value = 0

    def run(self):
        self.value += 1
        print("welcome to your work component")
        print("this is running inside a work")
