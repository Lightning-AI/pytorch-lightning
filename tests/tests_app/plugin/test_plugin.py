import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from lightning.app.plugin.plugin import _Run, _start_plugin_server
from lightning_cloud.openapi import Externalv1LightningappInstance


@pytest.fixture()
@mock.patch("lightning.app.plugin.plugin.uvicorn")
def mock_plugin_server(mock_uvicorn) -> TestClient:
    """This fixture returns a `TestClient` for the plugin server."""
    test_client = {}

    def create_test_client(app, **_):
        test_client["client"] = TestClient(app)

    mock_uvicorn.run.side_effect = create_test_client

    _start_plugin_server(8888)

    return test_client["client"]


@dataclass
class _MockResponse:
    content: bytes

    def raise_for_status(self):
        pass


def mock_requests_get(valid_url, return_value):
    """Used to replace `requests.get` with a function that returns the given value for the given valid URL and raises
    otherwise."""

    def inner(url):
        if url == valid_url:
            return _MockResponse(return_value)
        raise RuntimeError

    return inner


def as_tar_bytes(file_name, content):
    """Utility to encode the given string as a gzipped tar and return the bytes."""
    tar_fileobj = io.BytesIO()
    with tarfile.open(fileobj=tar_fileobj, mode="w|gz") as tar:
        content = content.encode("utf-8")
        tf = tarfile.TarInfo(file_name)
        tf.size = len(content)
        tar.addfile(tf, io.BytesIO(content))
    tar_fileobj.seek(0)
    return tar_fileobj.read()


_plugin_with_internal_error = """
from lightning.app.plugin.plugin import LightningPlugin

class TestPlugin(LightningPlugin):
    def run(self):
        raise RuntimeError("Internal Error")

plugin = TestPlugin()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the plugin server is only intended to run on linux.")
@pytest.mark.parametrize(
    ("body", "message", "tar_file_name", "content"),
    [
        (
            _Run(
                plugin_entrypoint="test",
                source_code_url="this_url_does_not_exist",
                project_id="any",
                cloudspace_id="any",
                cluster_id="any",
                plugin_arguments={},
                source_app="any",
                keep_machines_after_stop=False,
            ),
            "Error downloading plugin source:",
            None,
            b"",
        ),
        (
            _Run(
                plugin_entrypoint="test",
                source_code_url="http://test.tar.gz",
                project_id="any",
                cloudspace_id="any",
                cluster_id="any",
                plugin_arguments={},
                source_app="any",
                keep_machines_after_stop=False,
            ),
            "Error extracting plugin source:",
            None,
            b"this is not a tar",
        ),
        (
            _Run(
                plugin_entrypoint="plugin.py",
                source_code_url="http://test.tar.gz",
                project_id="any",
                cloudspace_id="any",
                cluster_id="any",
                plugin_arguments={},
                source_app="any",
                keep_machines_after_stop=False,
            ),
            "Error loading plugin:",
            "plugin.py",
            "this is not a plugin",
        ),
        (
            _Run(
                plugin_entrypoint="plugin.py",
                source_code_url="http://test.tar.gz",
                project_id="any",
                cloudspace_id="any",
                cluster_id="any",
                plugin_arguments={},
                source_app="any",
                keep_machines_after_stop=False,
            ),
            "Error running plugin:",
            "plugin.py",
            _plugin_with_internal_error,
        ),
    ],
)
@mock.patch("lightning.app.plugin.plugin.requests")
def test_run_errors(mock_requests, mock_plugin_server, body, message, tar_file_name, content):
    if tar_file_name is not None:
        content = as_tar_bytes(tar_file_name, content)

    mock_requests.get.side_effect = mock_requests_get("http://test.tar.gz", content)

    response = mock_plugin_server.post("/v1/runs", json=body.dict(exclude_none=True))

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert message in response.text


_plugin_with_job_run = """
from lightning.app.plugin.plugin import LightningPlugin

class TestPlugin(LightningPlugin):
    def run(self, name, entrypoint):
        return self.run_job(name, entrypoint)

plugin = TestPlugin()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the plugin server is only intended to run on linux.")
@mock.patch("lightning.app.runners.backends.cloud.CloudBackend")
@mock.patch("lightning.app.runners.cloud.CloudRuntime")
@mock.patch("lightning.app.plugin.plugin.requests")
def test_run_job(mock_requests, mock_cloud_runtime, mock_cloud_backend, mock_plugin_server):
    """Tests that running a job from a plugin calls the correct `CloudRuntime` methods with the correct arguments."""
    content = as_tar_bytes("plugin.py", _plugin_with_job_run)
    mock_requests.get.side_effect = mock_requests_get("http://test.tar.gz", content)

    body = _Run(
        plugin_entrypoint="plugin.py",
        source_code_url="http://test.tar.gz",
        project_id="test_project_id",
        cloudspace_id="test_cloudspace_id",
        cluster_id="test_cluster_id",
        plugin_arguments={"name": "test_name", "entrypoint": "test_entrypoint"},
        source_app="test_source_app",
        keep_machines_after_stop=True,
    )

    mock_app = mock.MagicMock()
    mock_cloud_runtime.load_app_from_file.return_value = mock_app
    mock_cloud_runtime.return_value.cloudspace_dispatch.return_value = Externalv1LightningappInstance(
        id="created_app_id"
    )

    response = mock_plugin_server.post("/v1/runs", json=body.dict(exclude_none=True))

    assert response.status_code == status.HTTP_200_OK, response.json()
    assert json.loads(response.text)["id"] == "created_app_id"

    mock_cloud_runtime.load_app_from_file.assert_called_once()
    assert "test_entrypoint" in mock_cloud_runtime.load_app_from_file.call_args[0][0]

    mock_cloud_runtime.assert_called_once_with(
        app=mock_app,
        entrypoint=Path("test_entrypoint"),
        start_server=True,
        env_vars={},
        secrets={},
        run_app_comment_commands=True,
        backend=mock.ANY,
    )

    mock_cloud_runtime().cloudspace_dispatch.assert_called_once_with(
        project_id=body.project_id,
        cloudspace_id=body.cloudspace_id,
        name="test_name",
        cluster_id=body.cluster_id,
        source_app=body.source_app,
        keep_machines_after_stop=body.keep_machines_after_stop,
    )


def test_healthz(mock_plugin_server):
    """Smoke test for the healthz endpoint."""
    assert mock_plugin_server.get("/healthz").status_code == 200
