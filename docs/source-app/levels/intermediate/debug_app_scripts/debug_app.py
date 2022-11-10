# app.py
import lightning as L
from lightning.app.runners import MultiProcessRuntime


class TrainComponent(L.LightningWork):
    def run(self, x):
        print(f'train a model on {x}')

class AnalyzeComponent(L.LightningWork):
    def run(self, x):
        print(f'analyze model on {x}')

class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent()
        self.analyze = AnalyzeComponent()

    def run(self):
        self.train.run("GPU machine 1")
        self.analyze.run("CPU machine 2")

app = L.LightningApp(WorkflowOrchestrator())
MultiProcessRuntime(app).dispatch()
