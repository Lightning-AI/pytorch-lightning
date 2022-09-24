import lightning as L
from lightning.app.testing.helpers import EmptyFlow, EmptyWork


class FlowB(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow_d = EmptyFlow()
        self.work_b = EmptyWork()

    def run(self):
        ...


class FlowA(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow_b = FlowB()
        self.flow_c = EmptyFlow()
        self.work_a = EmptyWork()

    def run(self):
        ...


app = L.LightningApp(FlowA())
