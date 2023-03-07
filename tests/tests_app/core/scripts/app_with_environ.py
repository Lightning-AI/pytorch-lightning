import lightning as L
import os


class MyWork(L.LightningWork):
    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute(name=os.environ.get("COMPUTE_NAME")))

    def run(self):
        pass


app = L.LightningApp(MyWork())
