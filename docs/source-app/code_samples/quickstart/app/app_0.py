from lightning.app import LightningWork, LightningFlow, LightningApp
from docs.quickstart.app_02 import HourLongWork


class RootFlow(LightningFlow):
    def __init__(self, child_work_1: LightningWork, child_work_2: LightningWork):
        super().__init__()
        self.child_work_1 = child_work_1
        self.child_work_2 = child_work_2

    def run(self):
        print(round(self.child_work_1.progress, 4), round(self.child_work_2.progress, 4))
        self.child_work_1.run()
        self.child_work_2.run()
        if self.child_work_1.progress == 1.0:
            print("1 hour later `child_work_1` started!")
        if self.child_work_2.progress == 1.0:
            print("1 hour later `child_work_2` started!")


app = LightningApp(RootFlow(HourLongWork(parallel=True), HourLongWork(parallel=True)))
