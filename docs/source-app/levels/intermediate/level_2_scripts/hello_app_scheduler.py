# app.py
from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute


class TrainComponent(LightningWork):
    def run(self, x):
        print(f'train a model on {x}')

class AnalyzeComponent(LightningWork):
    def run(self, x):
        print(f'analyze model on {x}')

class WorkflowOrchestrator(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent(cloud_compute=CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=CloudCompute('gpu'))

    def run(self):
        # run training once
        self.train.run("GPU machine 1")

        # run analysis once, then every hour again...
        if self.schedule("hourly"):
            self.analyze.run("CPU machine 2")

app = LightningApp(WorkflowOrchestrator())
