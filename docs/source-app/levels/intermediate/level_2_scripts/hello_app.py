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
        self.train.run("CPU machine 1")
        self.analyze.run("GPU machine 2")

app = LightningApp(WorkflowOrchestrator())
