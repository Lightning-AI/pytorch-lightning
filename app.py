from lightning.app import LightningFlow, LightningWork, LightningApp
from lightning.app.structures import Dict
class CounterWork(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False, parallel=True)
        self.counter = 0
    def run(self):
        self.counter += 1
        import time
        time.sleep(1)
        print("work", self.counter)


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.dict = Dict(**{"work_0": CounterWork(), "work_1": CounterWork()})

    def run(self):
        print("flow")
        for work_name, work in self.dict.items():
            work.run()
            if work_name == "work_0" and work.counter == 5:
                work.stop()

            if work.has_stopped:
                print("Starting a stopped work")
                work.run()

app = LightningApp(RootFlow())
