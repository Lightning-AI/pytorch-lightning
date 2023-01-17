from time import time

import lightning as L


class Work(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def run(self):
        while True:
            self.counter += 1
            time(1)


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = Work(cloud_compute=L.CloudCompute("gpu", preemptible=True), start_with_flow=False)

    def run(self):
        self.w.run()


app = L.LightningApp(Flow())
