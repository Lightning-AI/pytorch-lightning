# app.py
import subprocess
import lightning as L


class ExternalModelServer(L.LightningWork):
    def run(self, x):
        # compile
        process = subprocess.Popen('g++ model_server.cpp -o model_server')
        process.wait()
        process = subprocess.Popen('./model_server')
        process.wait()

class LocustLoadTester(L.LightningWork):
    def run(self, x):
        cmd = f'locust --master-host {self.host} --master-port {self.port}'
        process = subprocess.Popen(cmd)
        process.wait()

class WorkflowOrchestrator(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.serve = ExternalModelServer(
            cloud_compute=L.CloudCompute('cpu'), parallel=True
        )
        self.load_test = LocustLoadTester(cloud_compute=L.CloudCompute('cpu'))

    def run(self):
        # start the server (on a CPU machine 1)
        self.serve.run()

        # load testing when the server is up (on a separate cpu machine 2)
        if self.serve.state.RUNNING:
            self.load_test.run()

app = L.LightningApp(WorkflowOrchestrator())
