import os

from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import application_testing, LightningTestApp


class LightningTestMultiNodeApp(LightningTestApp):
    def on_before_run_once(self):
        res = super().on_before_run_once()
        if all(w.has_finished for w in self.works):
            return True
        return res


def test_multi_node_example():
    cwd = os.getcwd()
    new_cwd = os.path.join(_PROJECT_ROOT, "examples/app_multi_node")
    os.chdir(new_cwd)
    command_line = [
        "app.py",
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningTestMultiNodeApp, command_line)
    assert result.exit_code == 0
    os.chdir(cwd)
