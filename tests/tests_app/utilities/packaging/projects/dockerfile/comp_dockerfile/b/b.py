from lightning.app import LightningFlow


class BBB(LightningFlow):
    def __init__(self, work):
        super().__init__()
        self.work = work

    def run(self):
        self.stop()
