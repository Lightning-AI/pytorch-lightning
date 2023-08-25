import os

from lightning.app import CloudCompute, LightningApp, LightningWork


class MyWork(LightningWork):
    def __init__(self):
        super().__init__(cloud_compute=CloudCompute(name=os.environ.get("COMPUTE_NAME", "default")))

    def run(self):
        pass


app = LightningApp(MyWork())
