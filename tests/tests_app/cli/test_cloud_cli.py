import enum
import logging
import os
from dataclasses import dataclass
from functools import partial
from unittest import mock
from unittest.mock import ANY, MagicMock, call

import lightning.app.runners.backends.cloud as cloud_backend
import pytest
from click.testing import CliRunner
from lightning.app.cli.lightning_cli import run_app
from lightning.app.runners import cloud
from lightning.app.runners.cloud import CloudRuntime
from lightning_cloud.openapi import (
    V1CloudSpace,
    V1ListCloudSpacesResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)
from lightning_cloud.openapi.rest import ApiException

from tests_app import _PROJECT_ROOT

_FILE_PATH = os.path.join(_PROJECT_ROOT, "tests", "tests_app", "core", "scripts", "app_metadata.py")


@dataclass
class AppMetadata:
    id: str


@dataclass
class FakeResponse:
    lightningapps = [AppMetadata(id="my_app")]


class FakeLightningClient:
    def cloud_space_service_list_cloud_spaces(self, *args, **kwargs):
        return V1ListCloudSpacesResponse(cloudspaces=[])

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
        super().__init__()
        self.create_response = create_response

    def cloud_space_service_create_cloud_space(self, *args, **kwargs):
        return V1CloudSpace(id="my_app", name="app")

    def cloud_space_service_create_lightning_run(self, project_id, cloudspace_id, body):
        assert project_id == "test-project-id"
        return self.create_response

    def cloud_space_service_create_lightning_run_instance(self, project_id, cloudspace_id, id, body):
        assert project_id == "test-project-id"
        return self.create_response


@mock.patch("lightning.app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning.app.runners.runtime_type.CloudRuntime", CloudRuntimePatch)
@pytest.mark.parametrize("create_response", [RuntimeErrorResponse(), RuntimeErrorResponse2()])
def test_start_app(create_response, monkeypatch):
    monkeypatch.setattr(cloud, "V1LightningappInstanceState", MagicMock())
    monkeypatch.setattr(cloud, "CloudspaceIdRunsBody", MagicMock())
    monkeypatch.setattr(cloud, "V1Flowserver", MagicMock())
    monkeypatch.setattr(cloud, "V1LightningappInstanceSpec", MagicMock())
    monkeypatch.setattr(
        cloud_backend,
        "LightningClient",
        partial(FakeLightningClientCreate, create_response=create_response),
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

        cloud.CloudspaceIdRunsBody.assert_called_once()


class HttpHeaderDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reason = kwargs["reason"]
        self.status = kwargs["status"]
        self.data = kwargs["data"]

    def getheaders(self):
        return {}


class FakeLightningClientException(FakeLightningClient):
    def __init__(self, *args, message, **kwargs):
        super().__init__()
        self.message = message

    def cloud_space_service_list_cloud_spaces(self, *args, **kwargs):
        raise ApiException(
            http_resp=HttpHeaderDict(
                data=self.message,
                reason="",
                status=500,
            )
        )


@mock.patch("lightning.app.utilities.network.create_swagger_client", MagicMock())
@mock.patch("lightning.app.runners.runtime_type.CloudRuntime", CloudRuntimePatch)
@pytest.mark.parametrize(
    "message",
    [
        "Cannot create a new app, you have reached the maximum number (10) of apps. Either increase your quota or delete some of the existing apps"  # noqa E501
    ],
)
def test_start_app_exception(message, monkeypatch, caplog):
    monkeypatch.setattr(cloud, "V1LightningappInstanceState", MagicMock())
    monkeypatch.setattr(cloud, "CloudspaceIdRunsBody", MagicMock())
    monkeypatch.setattr(cloud, "V1Flowserver", MagicMock())
    monkeypatch.setattr(cloud, "V1LightningappInstanceSpec", MagicMock())
    monkeypatch.setattr(cloud, "LocalSourceCodeDir", MagicMock())
    monkeypatch.setattr(cloud, "_prepare_lightning_wheels_and_requirements", MagicMock())
    monkeypatch.setattr(cloud, "logger", logging.getLogger())

    runner = CliRunner()

    fake_grid_rest_client = partial(FakeLightningClientException, message=message)
    with caplog.at_level(logging.ERROR), mock.patch(
        "lightning.app.runners.backends.cloud.LightningClient", fake_grid_rest_client
    ):
        result = runner.invoke(run_app, [_FILE_PATH, "--cloud", "--open-ui=False"], catch_exceptions=False)
        assert result.exit_code == 1
    assert caplog.messages == [message]
