# app.py
from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.pdb import set_trace

class Component(LightningWork):
    def run(self, x):
        print(x)
        set_trace()

class WorkflowOrchestrator(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component = Component()

    def run(self):
        self.component.run('i love Lightning')

app = LightningApp(WorkflowOrchestrator())
