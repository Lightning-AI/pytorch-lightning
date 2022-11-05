# app.py
import lightning as L

class Component(L.LightningWork):
    def run(self, x):
        print(f'this string came from machine 0: " {x}')
        print('this string is on machine 1')

class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.component = Component()

    def run(self):
        x = 'this string is on machine 0'
        self.component(x)

app = L.LightningApp(WorkflowOrchestrator())