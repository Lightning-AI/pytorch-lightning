import os

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import application_testing, LightningTestApp


class LightningTestMultiNodeApp(LightningTestApp):
    def on_before_run_once(self):
        res = super().on_before_run_once()
        if self.works and all(w.has_stopped for w in self.works):
            assert len([w for w in self.works]) == 2
            return True
        return res


@pytest.mark.skipif(True, reason="flaky")
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


class LightningTestMultiNodeWorksApp(LightningTestApp):
    def on_before_run_once(self):
        res = super().on_before_run_once()
        if self.works and all(w.has_stopped for w in self.works):
            assert len([w for w in self.works]) == 2
            return True
        return res


def test_multi_node_example_2():
    cwd = os.getcwd()
    new_cwd = os.path.join(_PROJECT_ROOT, "examples/app_multi_node")
    os.chdir(new_cwd)
    command_line = [
        "app_work.py",
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningTestMultiNodeWorksApp, command_line)
    assert result.exit_code == 0
    os.chdir(cwd)
