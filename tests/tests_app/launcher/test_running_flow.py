import logging
import os
import signal
import sys
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import requests
from lightning.app.launcher import launcher, lightning_backend
from lightning.app.utilities.app_helpers import convert_print_to_logger_info
from lightning.app.utilities.enum import AppStage
from lightning.app.utilities.exceptions import ExitAppException


def _make_mocked_network_config(key, host):
    network_config = Mock()
    network_config.name = key
    network_config.host = host
    return network_config


@mock.patch("lightning.app.core.queues.QueuingSystem", mock.MagicMock())
@mock.patch("lightning.app.launcher.launcher.check_if_redis_running", MagicMock(return_value=True))
def test_running_flow(monkeypatch):
    app = MagicMock()
    flow = MagicMock()
    work = MagicMock()
    work.run.__name__ = "run"
    flow._layout = {}
    flow.name = "flowname"
    work.name = "workname"
    app.flows = [flow]
    flow.works.return_value = [work]

    def load_app_from_file(file):
        assert file == "file.py"
        return app

    class BackendMock:
        def __init__(self, return_value):
            self.called = 0
            self.return_value = return_value

        def _get_cloud_work_specs(self, *_):
            value = self.return_value if not self.called else []
            self.called += 1
            return value

    cloud_work_spec = Mock()
    cloud_work_spec.name = "workname"
    cloud_work_spec.spec.network_config = [
        _make_mocked_network_config("key1", "x.lightning.ai"),
    ]
    monkeypatch.setattr(launcher, "load_app_from_file", load_app_from_file)
    monkeypatch.setattr(launcher, "start_server", MagicMock())
    monkeypatch.setattr(lightning_backend, "LightningClient", MagicMock())
    lightning_backend.CloudBackend._get_cloud_work_specs = BackendMock(
        return_value=[cloud_work_spec]
    )._get_cloud_work_specs
    monkeypatch.setattr(lightning_backend.CloudBackend, "_get_project_id", MagicMock())
    monkeypatch.setattr(lightning_backend.CloudBackend, "_get_app_id", MagicMock())
    queue_system = MagicMock()
    queue_system.REDIS = MagicMock()
    monkeypatch.setattr(launcher, "QueuingSystem", queue_system)
    monkeypatch.setattr(launcher, "StorageOrchestrator", MagicMock())

    response = MagicMock()
    response.status_code = 200
    monkeypatch.setattr(requests, "get", MagicMock(return_value=response))

    # testing with correct base URL
    with pytest.raises(SystemExit, match="0"):
        launcher.run_lightning_flow("file.py", queue_id="", base_url="http://localhost:8080")
    assert flow._layout["target"] == "http://localhost:8080/flowname/"

    app._run.assert_called_once()

    # testing with invalid base URL
    with pytest.raises(ValueError, match="Base URL doesn't have a valid scheme"):
        launcher.run_lightning_flow("file.py", queue_id="", base_url="localhost:8080")

    app.flows = []

    def run_patch():
        raise Exception

    app._run = run_patch

    with pytest.raises(SystemExit, match="1"):
        launcher.run_lightning_flow("file.py", queue_id="", base_url="localhost:8080")

    def run_patch():
        app.stage = AppStage.FAILED

    app._run = run_patch

    with pytest.raises(SystemExit, match="1"):
        launcher.run_lightning_flow("file.py", queue_id="", base_url="localhost:8080")

    def run_patch():
        raise ExitAppException

    if sys.platform == "win32":
        return

    app.stage = AppStage.STOPPING

    app._run = run_patch
    with pytest.raises(SystemExit, match="0"):
        launcher.run_lightning_flow("file.py", queue_id="", base_url="localhost:8080")

    def run_method():
        os.kill(os.getpid(), signal.SIGTERM)

    app._run = run_method
    monkeypatch.setattr(lightning_backend.CloudBackend, "resolve_url", MagicMock())
    with pytest.raises(SystemExit, match="0"):
        launcher.run_lightning_flow("file.py", queue_id="", base_url="localhost:8080")
    assert app.stage == AppStage.STOPPING


def test_replace_print_to_info(caplog, monkeypatch):
    monkeypatch.setattr("lightning.app._logger", logging.getLogger())

    @convert_print_to_logger_info
    def fn_captured(value):
        print(value)

    with caplog.at_level(logging.INFO):
        fn_captured(1)

    assert caplog.messages == ["1"]
