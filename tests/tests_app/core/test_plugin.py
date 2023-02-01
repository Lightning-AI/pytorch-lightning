from pathlib import Path
from unittest import mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from lightning.app.core.plugin import _Run, _start_plugin_server, Plugin


@pytest.fixture()
@mock.patch("lightning.app.core.plugin.uvicorn")
def mock_plugin_server(mock_uvicorn) -> TestClient:
    """This fixture returns a `TestClient` for the plugin server."""

    test_client = {}

    def create_test_client(app, **_):
        test_client["client"] = TestClient(app)

    mock_uvicorn.run.side_effect = create_test_client

    _start_plugin_server("0.0.0.0", 8888)

    return test_client["client"]


def test_run_bad_request(mock_plugin_server):
    body = _Run(
        plugin_name="test",
        project_id="any",
        cloudspace_id="any",
        name="any",
        entrypoint="any",
    )

    response = mock_plugin_server.post("/v1/runs", json=body.dict(exclude_none=True))

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "App ID must be specified" in response.text


@mock.patch("lightning.app.runners.cloud.CloudRuntime")
def test_run_app(mock_cloud_runtime, mock_plugin_server):
    """Tests that app dispatch call the correct `CloudRuntime` methods with the correct arguments."""
    body = _Run(
        plugin_name="app",
        project_id="test_project_id",
        cloudspace_id="test_cloudspace_id",
        name="test_name",
        entrypoint="test_entrypoint",
    )

    mock_app = mock.MagicMock()
    mock_cloud_runtime.load_app_from_file.return_value = mock_app

    response = mock_plugin_server.post("/v1/runs", json=body.dict(exclude_none=True))

    assert response.status_code == status.HTTP_200_OK

    mock_cloud_runtime.load_app_from_file.assert_called_once_with("/content/test_entrypoint")

    mock_cloud_runtime.assert_called_once_with(
        app=mock_app,
        entrypoint=Path("/content/test_entrypoint"),
        start_server=True,
        env_vars={},
        secrets={},
        run_app_comment_commands=True,
    )

    mock_cloud_runtime().cloudspace_dispatch.assert_called_once_with(
        project_id=body.project_id,
        cloudspace_id=body.cloudspace_id,
        name=body.name,
        cluster_id=body.cluster_id,
    )


@mock.patch("lightning.app.utilities.commands.base._download_command")
@mock.patch("lightning.app.utilities.cli_helpers._LightningAppOpenAPIRetriever")
def test_run_plugin(mock_retriever, mock_download_command, mock_plugin_server):
    """Tests that running a plugin calls the correct `CloudRuntime` methods with the correct arguments."""
    body = _Run(
        plugin_name="test_plugin",
        project_id="test_project_id",
        cloudspace_id="test_cloudspace_id",
        name="test_name",
        entrypoint="test_entrypoint",
        app_id="test_app_id",
    )

    mock_plugin = mock.MagicMock(spec=Plugin)
    mock_download_command.return_value = mock_plugin

    mock_retriever.return_value.api_commands = {
        body.plugin_name: {"cls_path": "test_cls_path", "cls_name": "test_cls_name"}
    }

    response = mock_plugin_server.post("/v1/runs", json=body.dict(exclude_none=True))

    assert response.status_code == status.HTTP_200_OK

    mock_retriever.assert_called_once_with(body.app_id)

    mock_download_command.assert_called_once_with(
        body.plugin_name,
        "test_cls_path",
        "test_cls_name",
        body.app_id,
        target_file=mock.ANY,
    )

    mock_plugin._setup.assert_called_once_with(app_id=body.app_id)
    mock_plugin.run.assert_called_once_with(body.name, body.entrypoint)
