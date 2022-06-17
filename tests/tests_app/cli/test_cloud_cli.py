import enum
import logging
import os
from dataclasses import dataclass
from functools import partial
from unittest import mock
from unittest.mock import ANY, call, MagicMock

import pytest
from click.testing import CliRunner
from lightning_cloud.openapi import (
    V1LightningappV2,
    V1ListLightningappInstancesResponse,
    V1ListLightningappsV2Response,
    V1ListMembershipsResponse,
    V1Membership,
)
from lightning_cloud.openapi.rest import ApiException

import lightning_app.runners.backends.cloud as cloud_backend
from lightning_app import _PROJECT_ROOT
from lightning_app.cli.lightning_cli import run_app
from lightning_app.runners import cloud
from lightning_app.runners.cloud import CloudRuntime

_FILE_PATH = os.path.join(_PROJECT_ROOT, "tests", "core", "scripts", "app_metadata.py")


@dataclass
class AppMetadata:
    id: str


@dataclass
class FakeResponse:
    lightningapps = [AppMetadata(id="my_app")]


class FakeLightningClient:
    def __init__(self, response, api_client=None):
        self._response = response

    def lightningapp_instance_service_list_lightningapp_instances(self, *args, **kwargs):
        return V1ListLightningappInstancesResponse(lightningapps=[])

    def lightningapp_service_delete_lightningapp(self, id: str = None):
        assert id == "my_app"

    def projects_service_list_memberships(self):
        return V1ListMembershipsResponse(memberships=[V1Membership(name="test-project", project_id="test-project-id")])


class CloudRuntimePatch(CloudRuntime):
    def __init__(self, *args, **kwargs):
        super_init = super().__init__
        if hasattr(super_init, "__wrapped__"):
            super_init.__wrapped__(self, *args, **kwargs)
        else:
            super_init(*args, **kwargs)


class V1LightningappInstanceState(enum.Enum):
    FAILED = "failed"
    SUCCESS = "success"


@dataclass
class FailedStatus:
    phase = V1LightningappInstanceState.FAILED


@dataclass
class SuccessStatus:
    phase = V1LightningappInstanceState.SUCCESS


@dataclass
class RuntimeErrorResponse:
    id = "my_app"
    source_upload_url = "something"
    status = FailedStatus()


@dataclass
class RuntimeErrorResponse2:
    id = "my_app"
    source_upload_url = ""
    status = SuccessStatus()


@dataclass
class SuccessResponse:
    id = "my_app"
    source_upload_url = "something"
    status = SuccessStatus()


@dataclass
class ExceptionResponse:
    status = FailedStatus()


class FakeLightningClientCreate(FakeLightningClient):
    def __init__(self, *args, create_response, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_response = create_response

    def lightningapp_v2_service_list_lightningapps_v2(self, *args, **kwargs):
        return V1ListLightningappsV2Response(lightningapps=[V1LightningappV2(id="my_app", name="app")])

    def lightningapp_v2_service_create_lightningapp_release(self, project_id, app_id, body):
        assert project_id == "test-project-id"
        return self.create_response

    def lightningapp_v2_service_create_lightningapp_release_instance(self, project_id, app_id, release_id, body):
        assert project_id == "test-project-id"
        return self.create_response


@mock.patch("lightning_app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning_app.runners.runtime_type.CloudRuntime", CloudRuntimePatch)
@pytest.mark.parametrize("create_response", [RuntimeErrorResponse(), RuntimeErrorResponse2()])
def test_start_app(create_response, monkeypatch):

    monkeypatch.setattr(cloud, "V1LightningappInstanceState", MagicMock())
    monkeypatch.setattr(cloud, "Body8", MagicMock())
    monkeypatch.setattr(cloud, "V1Flowserver", MagicMock())
    monkeypatch.setattr(cloud, "V1LightningappInstanceSpec", MagicMock())
    monkeypatch.setattr(
        cloud_backend,
        "LightningClient",
        partial(FakeLightningClientCreate, response=FakeResponse(), create_response=create_response),
    )
    monkeypatch.setattr(cloud, "LocalSourceCodeDir", MagicMock())
    monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", MagicMock())

    runner = CliRunner()

    def run():
        result = runner.invoke(run_app, [_FILE_PATH, "--cloud", "--open-ui=False"], catch_exceptions=False)
        assert result.exit_code == 0

    if isinstance(create_response, RuntimeErrorResponse):
        cloud.V1LightningappInstanceState.FAILED = V1LightningappInstanceState.FAILED
        with pytest.raises(RuntimeError, match="Failed to create the application"):
            run()
    elif isinstance(create_response, RuntimeErrorResponse2):
        with pytest.raises(RuntimeError, match="The source upload url is empty."):
            run()
    elif isinstance(create_response, RuntimeErrorResponse2):
        with pytest.raises(RuntimeError, match="The source upload url is empty."):
            run()
    else:
        run()
        mocks_calls = cloud.LocalSourceCodeDir._mock_mock_calls
        assert len(mocks_calls) == 5
        assert str(mocks_calls[0].kwargs["path"]) == os.path.dirname(_FILE_PATH)
        mocks_calls[1].assert_called_once()
        mocks_calls[2].assert_called_once(url="url")

        assert cloud.V1Flowserver._mock_call_args_list == [call(name="root.flow_b")]

        cloud.V1LightningappInstanceSpec._mock_call_args.assert_called_once(
            app_entrypoint_file=_FILE_PATH,
            enable_app_server=True,
            works=ANY,
            flow_servers=ANY,
        )

        cloud.Body8.assert_called_once()


class FakeLightningClientException(FakeLightningClient):
    def __init__(self, *args, message, api_client=None, **kwargs):
        super().__init__(*args, api_client=api_client, **kwargs)
        self.message = message

    def lightningapp_v2_service_list_lightningapps_v2(self, *args, **kwargs):
        class HttpHeaderDict(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.reason = ""
                self.status = 500
                self.data = kwargs["data"]

            def getheaders(self):
                return {}

        raise ApiException(
            http_resp=HttpHeaderDict(
                data=self.message,
                reason="",
                status=500,
            )
        )


@mock.patch("lightning_app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning_app.runners.runtime_type.CloudRuntime", CloudRuntimePatch)
@pytest.mark.parametrize(
    "message",
    [
        "Cannot create a new app, you have reached the maximum number (10) of apps. Either increase your quota or delete some of the existing apps"  # noqa E501
    ],
)
def test_start_app_exception(message, monkeypatch, caplog):

    monkeypatch.setattr(cloud, "V1LightningappInstanceState", MagicMock())
    monkeypatch.setattr(cloud, "Body8", MagicMock())
    monkeypatch.setattr(cloud, "V1Flowserver", MagicMock())
    monkeypatch.setattr(cloud, "V1LightningappInstanceSpec", MagicMock())
    monkeypatch.setattr(cloud, "LocalSourceCodeDir", MagicMock())
    monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", MagicMock())
    monkeypatch.setattr(cloud, "logger", logging.getLogger())

    runner = CliRunner()

    fake_grid_rest_client = partial(FakeLightningClientException, response=FakeResponse(), message=message)
    with caplog.at_level(logging.ERROR):
        with mock.patch("lightning_app.runners.backends.cloud.LightningClient", fake_grid_rest_client):
            result = runner.invoke(run_app, [_FILE_PATH, "--cloud", "--open-ui=False"], catch_exceptions=False)
            assert result.exit_code == 1
    assert caplog.messages == [message]
