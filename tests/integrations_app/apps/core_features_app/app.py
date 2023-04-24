import os

from lightning.app.core import LightningApp, LightningFlow


class EnvVarTestApp(LightningFlow):
    def __init__(self):
        super().__init__()

    def run(self):
        # these env vars are set here: tests/integrations_app/test_core_features_app.py:15
        assert os.getenv("FOO", "") == "bar"
        assert os.getenv("BLA", "") == "bloz"
        self.stop()


app = LightningApp(EnvVarTestApp())
