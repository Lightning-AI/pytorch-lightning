from lightning.app import LightningFlow, LightningApp, CloudCompute
from lightning.app.components import TracerPythonScript


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.runner = TracerPythonScript(
            "train.py",
            cloud_compute=CloudCompute("gpu"),
        )

    def run(self):
        self.runner.run()


app = LightningApp(RootFlow())
