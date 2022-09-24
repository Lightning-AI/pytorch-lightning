import lightning as L


class ComponentA(L.LightningFlow):
    def run(self):
        print("hello from component A")
