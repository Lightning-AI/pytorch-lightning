import os
import sys
from unittest import mock

import pytest
from tests_examples_app.public import _PATH_EXAMPLES

from lightning_app.testing.testing import application_testing, LightningTestApp


class LightningTestMultiNodeApp(LightningTestApp):
    def on_before_run_once(self):
        res = super().on_before_run_once()
        if self.works and all(w.has_stopped for w in self.works):
            assert len([w for w in self.works]) == 2
            return True
        return res


@pytest.mark.skip(reason="flaky")
@mock.patch("lightning_app.components.multi_node.base.is_running_in_cloud", return_value=True)
def test_multi_node_example(_, monkeypatch):
    monkeypatch.chdir(os.path.join(_PATH_EXAMPLES, "app_multi_node"))
    command_line = [
        "app.py",
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningTestMultiNodeApp, command_line)
    assert result.exit_code == 0


class LightningTestMultiNodeWorksApp(LightningTestApp):
    def on_before_run_once(self):
        res = super().on_before_run_once()
        if self.works and all(w.has_stopped for w in self.works):
            assert len([w for w in self.works]) == 2
            return True
        return res


@pytest.mark.parametrize(
    "app_name",
    [
        "train_pytorch.py",
        "train_any.py",
        # "app_lite_work.py",
        "train_pytorch_spawn.py",
        # "app_pl_work.py": TODO Add once https://github.com/Lightning-AI/lightning/issues/15556 is resolved.
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason="flaky")
@mock.patch("lightning_app.components.multi_node.base.is_running_in_cloud", return_value=True)
def test_multi_node_examples(_, app_name, monkeypatch):
    monkeypatch.chdir(os.path.join(_PATH_EXAMPLES, "app_multi_node"))
    command_line = [
        app_name,
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningTestMultiNodeWorksApp, command_line)
    assert result.exit_code == 0
