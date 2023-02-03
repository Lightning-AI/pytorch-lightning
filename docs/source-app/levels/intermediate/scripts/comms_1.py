# app.py
import lightning as L

class Component(L.LightningWork):
    def run(self, x):
        print(f'MACHINE 1: this string came from machine 0: "{x}"')
        print('MACHINE 1: this string is on machine 1')

class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component = Component()

    def run(self):
        x = 'hello from machine 0'
        self.component.run(x)

app = L.LightningApp(WorkflowOrchestrator())
