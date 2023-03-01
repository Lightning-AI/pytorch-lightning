# app.py
import lightning as L


class TrainComponent(L.LightningWork):
    def run(self, message):
        for i in range(100000000000):
            print(message, i)

class AnalyzeComponent(L.LightningWork):
    def run(self, message):
        for i in range(100000000000):
            print(message, i)

class LitWorkflow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent(cloud_compute=L.CloudCompute('cpu'))
        self.baseline_1 = TrainComponent(cloud_compute=L.CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=L.CloudCompute('cpu'))

    def run(self):
        self.train.run("machine A counting")
        self.baseline_1.run("machine C counting")
        self.analyze.run("machine B counting")

app = L.LightningApp(LitWorkflow())
