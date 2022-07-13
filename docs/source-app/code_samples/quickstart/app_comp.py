from lightning_app import LightningApp, LightningFlow
from lightning_app.testing.helpers import EmptyFlow, EmptyWork


class FlowB(LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow_d = EmptyFlow()
        self.work_b = EmptyWork()

    def run(self):
        ...


class FlowA(LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow_b = FlowB()
        self.flow_c = EmptyFlow()
        self.work_a = EmptyWork()

    def run(self):
        ...


app = LightningApp(FlowA())
