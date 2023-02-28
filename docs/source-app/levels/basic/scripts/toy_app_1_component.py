# app.py
import lightning as L

class Component(L.LightningWork):
    def run(self, x):
        print(x)


class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component = Component()

    def run(self):
        self.component.run('i love Lightning')

app = L.LightningApp(WorkflowOrchestrator())
