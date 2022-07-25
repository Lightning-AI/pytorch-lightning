import lightning as L
from lightning.app.components.python import TracerPythonScript


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.runner = TracerPythonScript(
            "train.py",
            cloud_compute=L.CloudCompute("gpu"),
        )

    def run(self):
        self.runner.run()


app = L.LightningApp(RootFlow())
