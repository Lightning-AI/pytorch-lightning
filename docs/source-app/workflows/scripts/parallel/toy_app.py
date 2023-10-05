# app.py
from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute


class TrainComponent(LightningWork):
    def run(self, message):
        for i in range(100000000000):
            print(message, i)

class AnalyzeComponent(LightningWork):
    def run(self, message):
        for i in range(100000000000):
            print(message, i)

class LitWorkflow(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent(cloud_compute=CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=CloudCompute('cpu'))


    def run(self):
        self.train.run("machine A counting")
        self.analyze.run("machine B counting")


app = LightningApp(LitWorkflow())
