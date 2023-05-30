from lightning.app import LightningFlow


class B(LightningFlow):
    def __init__(self, work):
        super().__init__()
        self.work = work

    def run(self):
        self.stop()
