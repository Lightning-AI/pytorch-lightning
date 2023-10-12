import os
from pathlib import Path

from lightning.app import LightningApp, LightningFlow

from examples.components.python.component_tracer import PLTracerPythonScript


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        script_path = Path(__file__).parent / "pl_script.py"
        self.tracer_python_script = PLTracerPythonScript(script_path)

    def run(self):
        assert os.getenv("GLOBAL_RANK", "0") == "0"
        if not self.tracer_python_script.has_started:
            self.tracer_python_script.run()
        if self.tracer_python_script.has_succeeded:
            self.stop("tracer script succeed")
        if self.tracer_python_script.has_failed:
            self.stop("tracer script failed")


app = LightningApp(RootFlow())
