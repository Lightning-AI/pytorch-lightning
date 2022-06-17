import os
from unittest import mock
from unittest.mock import ANY, MagicMock

import pytest

import lightning_app
from lightning_app import LightningFlow
from lightning_app.frontend.web import healthz, StaticWebFrontend
from lightning_app.storage.path import storage_root_dir


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


@mock.patch("lightning_app.frontend.web.mp.Process")
def test_start_stop_server_through_frontend(process_mock):
    frontend = StaticWebFrontend(serve_dir=".")
    frontend.flow = MockFlow()
    frontend.start_server("localhost", 5000)
    log_file_root = storage_root_dir()
    process_mock.assert_called_once_with(
        target=lightning_app.frontend.web.start_server,
        kwargs={
            "host": "localhost",
            "port": 5000,
            "serve_dir": ".",
            "path": "/root.my.flow",
            "log_file": os.path.join(log_file_root, "frontend", "logs.log"),
        },
    )
    process_mock().start.assert_called_once()
    frontend.stop_server()
    process_mock().kill.assert_called_once()


@mock.patch("lightning_app.frontend.web.uvicorn")
def test_start_server_through_function(uvicorn_mock, tmpdir, monkeypatch):
    FastAPIMock = MagicMock()
    FastAPIMock.mount = MagicMock()
    FastAPIGetDecoratorMock = MagicMock()
    FastAPIMock.get.return_value = FastAPIGetDecoratorMock
    monkeypatch.setattr(lightning_app.frontend.web, "FastAPI", MagicMock(return_value=FastAPIMock))

    lightning_app.frontend.web.start_server(serve_dir=tmpdir, host="myhost", port=1000, path="/test-flow")
    uvicorn_mock.run.assert_called_once_with(app=ANY, host="myhost", port=1000, log_config=ANY)
    FastAPIMock.mount.assert_called_once_with("/test-flow", ANY, name="static")
    FastAPIMock.get.assert_called_once_with("/test-flow/healthz", status_code=200)

    FastAPIGetDecoratorMock.assert_called_once_with(healthz)

    # path has default value "/"
    FastAPIMock.mount = MagicMock()
    lightning_app.frontend.web.start_server(serve_dir=tmpdir, host="myhost", port=1000)
    FastAPIMock.mount.assert_called_once_with("/", ANY, name="static")


def test_healthz():
    assert healthz() == {"status": "ok"}


@mock.patch("lightning_app.frontend.web.uvicorn")
def test_start_server_find_free_port(uvicorn_mock, tmpdir):
    lightning_app.frontend.web.start_server(serve_dir=tmpdir, host="myhost")
    assert uvicorn_mock.run.call_args_list[0].kwargs["port"] > 0
