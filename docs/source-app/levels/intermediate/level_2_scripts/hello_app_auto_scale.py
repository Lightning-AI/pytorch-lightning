# app.py
import lightning as L


class TrainComponent(L.LightningWork):
    def run(self, x):
        print(f'train a model on {x}')

class AnalyzeComponent(L.LightningWork):
    def run(self, x):
        print(f'analyze model on {x}')

class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent(cloud_compute=L.CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=L.CloudCompute('gpu'))

    def run(self):
        # run() starts the machine
        self.train.run("GPU machine 1")

        # stop() stops the machine
        self.train.stop()

        # run analysis ONLY when machine 1 stopped
        if self.train.status.STOPPED:
            self.analyze.run("CPU machine 2")

app = L.LightningApp(WorkflowOrchestrator())
