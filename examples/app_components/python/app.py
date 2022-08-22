import os
from pathlib import Path

import lightning as L
from examples.components.python.component_tracer import PLTracerPythonScript


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        script_path = Path(__file__).parent / "pl_script.py"
        self.tracer_python_script = PLTracerPythonScript(script_path)

    def run(self):
        assert os.getenv("GLOBAL_RANK", "0") == "0"
        if not self.tracer_python_script.has_started:
            self.tracer_python_script.run()
        if self.tracer_python_script.has_succeeded:
            self._exit("tracer script succeed")
        if self.tracer_python_script.has_failed:
            self._exit("tracer script failed")


app = L.LightningApp(RootFlow())
