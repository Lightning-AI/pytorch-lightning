import os

from lightning_app.core import LightningApp, LightningFlow


class EnvVarTestApp(LightningFlow):
    def __init__(self):
        super().__init__()

    def run(self):
        # these env vars are set here: tests/tests_app_examples/test_core_features_app.py:15
        assert os.getenv("FOO", "") == "bar"
        assert os.getenv("BLA", "") == "bloz"
        self._exit()


app = LightningApp(EnvVarTestApp())
