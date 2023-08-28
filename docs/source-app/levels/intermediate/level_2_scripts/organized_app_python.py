# app.py
import subprocess
from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute


class ExternalModelServer(LightningWork):
    def run(self, x):
        # compile
        process = subprocess.Popen('g++ model_server.cpp -o model_server')
        process.wait()
        process = subprocess.Popen('./model_server')
        process.wait()

class LocustLoadTester(LightningWork):
    def run(self, x):
        cmd = f'locust --master-host {self.host} --master-port {self.port}'
        process = subprocess.Popen(cmd)
        process.wait()

class WorkflowOrchestrator(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.serve = ExternalModelServer(
            cloud_compute=CloudCompute('cpu'), parallel=True
        )
        self.load_test = LocustLoadTester(cloud_compute=CloudCompute('cpu'))

    def run(self):
        # start the server (on a CPU machine 1)
        self.serve.run()

        # load testing when the server is up (on a separate cpu machine 2)
        if self.serve.state.RUNNING:
            self.load_test.run()

app = LightningApp(WorkflowOrchestrator())
