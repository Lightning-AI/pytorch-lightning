from lightning_app import LightningFlow
from lightning_app.core.app import LightningApp
from lightning_app.runners import SingleProcessRuntime


class Flow(LightningFlow):
    def run(self):
        raise KeyboardInterrupt


def on_before_run():
    pass


def test_single_process_runtime(tmpdir):

    app = LightningApp(Flow())
    SingleProcessRuntime(app, start_server=False).dispatch(on_before_run=on_before_run)
