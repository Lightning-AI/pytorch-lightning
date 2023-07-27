# app.py
import lightning as L
from lightning.app.pdb import set_trace

class Component(L.LightningWork):
    def run(self, x):
        print(x)
        set_trace()

class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component = Component()

    def run(self):
        self.component.run('i love Lightning')

app = L.LightningApp(WorkflowOrchestrator())
