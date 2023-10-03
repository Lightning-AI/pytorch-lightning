from time import sleep

from lightning.app import LightningWork, LightningFlow, LightningApp


class HourLongWork(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False)
        self.progress = 0.0

    def run(self):
        self.progress = 0.0
        for _ in range(3600):
            self.progress += 1.0 / 3600
            sleep(1)


class RootFlow(LightningFlow):
    def __init__(self, child_work: LightningWork):
        super().__init__()
        self.child_work = child_work

    def run(self):
        # prints the progress from the child work
        print(round(self.child_work.progress, 4))
        self.child_work.run()
        if self.child_work.counter == 1.0:
            print("1 hour later!")


app = LightningApp(RootFlow(HourLongWork()))
