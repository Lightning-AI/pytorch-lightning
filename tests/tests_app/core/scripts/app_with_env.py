import os

import lightning as L


class MyWork(L.LightningWork):
    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute(name=os.environ.get("COMPUTE_NAME", "default")))

    def run(self):
        pass


app = L.LightningApp(MyWork())
