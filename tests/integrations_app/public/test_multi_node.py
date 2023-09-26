import os
from unittest import mock

import pytest
from lightning.app.testing.helpers import _RunIf
from lightning.app.testing.testing import LightningTestApp, application_testing
from lightning_utilities.core.imports import package_available

from integrations_app.public import _PATH_EXAMPLES


class LightningTestMultiNodeApp(LightningTestApp):
    def on_before_run_once(self):
        res = super().on_before_run_once()
        if self.works and all(w.has_stopped for w in self.works):
            assert len(self.works) == 2
            return True
        return res


# for the skip to work, the package needs to be installed without editable mode
_SKIP_LIGHTNING_UNAVAILABLE = pytest.mark.skipif(not package_available("lightning"), reason="script requires lightning")


@pytest.mark.parametrize(
    "app_name",
    [
        "train_pytorch.py",
        "train_any.py",
        "train_pytorch_spawn.py",
        pytest.param("train_fabric.py", marks=_SKIP_LIGHTNING_UNAVAILABLE),
        pytest.param("train_lt_script.py", marks=_SKIP_LIGHTNING_UNAVAILABLE),
        pytest.param("train_lt.py", marks=_SKIP_LIGHTNING_UNAVAILABLE),
    ],
)
@_RunIf(skip_windows=True)  # flaky
@mock.patch("lightning.app.components.multi_node.base.is_running_in_cloud", return_value=True)
def test_multi_node_examples(_, app_name, monkeypatch):
    # note: this test will fail locally:
    # * if you installed `lightning.app`, then the examples need to be
    #   rewritten to use `lightning.app` imports (CI does this)
    # * if you installed `lightning`, then the imports in this file and mocks
    #   need to be changed to use `lightning`.
    monkeypatch.chdir(os.path.join(_PATH_EXAMPLES, "multi_node"))
    command_line = [app_name, "--blocking", "False", "--open-ui", "False", "--setup"]
    result = application_testing(LightningTestMultiNodeApp, command_line)
    assert result.exit_code == 0
