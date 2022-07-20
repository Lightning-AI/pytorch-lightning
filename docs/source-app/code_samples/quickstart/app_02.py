from time import sleep

import lightning as L


# This work takes an hour to run
class HourLongWork(L.LightningWork):
    def __init__(self, parallel: bool = False):
        super().__init__(parallel=parallel)
        self.progress = 0.0

    def run(self):
        self.progress = 0.0
        for _ in range(3600):
            self.progress += 1.0 / 3600  # Reporting my progress to the Flow.
            sleep(1)


class RootFlow(L.LightningFlow):
    def __init__(self, child_work: L.LightningWork):
        super().__init__()
        self.child_work = child_work

    def run(self):
        # prints the progress from the child work
        print(round(self.child_work.progress, 4))
        self.child_work.run()
        if self.child_work.counter == 1.0:
            print("1 hour later!")


app = L.LightningApp(RootFlow(HourLongWork()))
