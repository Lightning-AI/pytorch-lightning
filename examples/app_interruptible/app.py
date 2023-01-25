from time import sleep

import lightning as L


class Work(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def run(self):
        while True:
            print(self.counter)
            self.counter += 1
            sleep(1)


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = Work(
            cloud_compute=L.CloudCompute("gpu", interruptible=True),
            start_with_flow=False,
            parallel=True,
        )

    def run(self):
        self.w.run()
        print(self.w.counter)


app = L.LightningApp(Flow())
