import lightning as L


class ComponentB(L.LightningFlow):
    def run(self):
        print("hello from component B")
