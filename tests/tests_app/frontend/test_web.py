import os
from unittest import mock
from unittest.mock import ANY, MagicMock

import lightning.app
import pytest
from lightning.app import LightningFlow
from lightning.app.frontend.web import StaticWebFrontend, _healthz
from lightning.app.storage.path import _storage_root_dir


def test_stop_server_not_running():
    frontend = StaticWebFrontend(serve_dir=".")
    with pytest.raises(RuntimeError, match="Server is not running."):
        frontend.stop_server()


class MockFlow(LightningFlow):
    @property
    def name(self):
        return "root.my.flow"

    def run(self):
        pass


@mock.patch("lightning.app.frontend.web.mp.Process")
def test_start_stop_server_through_frontend(process_mock):
    frontend = StaticWebFrontend(serve_dir=".")
    frontend.flow = MockFlow()
    frontend.start_server("localhost", 5000)
    log_file_root = _storage_root_dir()
    process_mock.assert_called_once_with(
        target=lightning.app.frontend.web._start_server,
        kwargs={
            "host": "localhost",
            "port": 5000,
            "serve_dir": ".",
            "path": "/root.my.flow",
            "log_file": os.path.join(log_file_root, "frontend", "logs.log"),
            "root_path": "",
        },
    )
    process_mock().start.assert_called_once()
    frontend.stop_server()
    process_mock().kill.assert_called_once()


@mock.patch("lightning.app.frontend.web.uvicorn")
@pytest.mark.parametrize("root_path", ["", "/base"])
def test_start_server_through_function(uvicorn_mock, tmpdir, monkeypatch, root_path):
    FastAPIMock = MagicMock()
    FastAPIMock.mount = MagicMock()
    FastAPIGetDecoratorMock = MagicMock()
    FastAPIMock.get.return_value = FastAPIGetDecoratorMock
    monkeypatch.setattr(lightning.app.frontend.web, "FastAPI", MagicMock(return_value=FastAPIMock))

    lightning.app.frontend.web._start_server(
        serve_dir=tmpdir, host="myhost", port=1000, path="/test-flow", root_path=root_path
    )
    uvicorn_mock.run.assert_called_once_with(app=ANY, host="myhost", port=1000, log_config=ANY, root_path=root_path)

    FastAPIMock.mount.assert_called_once_with(root_path or "/test-flow", ANY, name="static")
    FastAPIMock.get.assert_called_once_with("/test-flow/healthz", status_code=200)

    FastAPIGetDecoratorMock.assert_called_once_with(_healthz)

    # path has default value "/"
    FastAPIMock.mount = MagicMock()
    lightning.app.frontend.web._start_server(serve_dir=tmpdir, host="myhost", port=1000, root_path=root_path)
    FastAPIMock.mount.assert_called_once_with(root_path or "/", ANY, name="static")


def test_healthz():
    assert _healthz() == {"status": "ok"}


@mock.patch("lightning.app.frontend.web.uvicorn")
def test_start_server_find_free_port(uvicorn_mock, tmpdir):
    lightning.app.frontend.web._start_server(serve_dir=tmpdir, host="myhost")
    assert uvicorn_mock.run.call_args_list[0].kwargs["port"] > 0
