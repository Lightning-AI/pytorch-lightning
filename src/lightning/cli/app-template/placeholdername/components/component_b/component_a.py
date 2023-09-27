from lightning.app import LightningFlow


class ComponentB(LightningFlow):
    def run(self):
        print("hello from component B")
