import os
from unittest import mock

from lightning.app.utilities.cloud import _get_project, is_running_in_cloud
from lightning_cloud.openapi.models import V1Project


def test_get_project_queries_by_project_id_directly_if_it_is_passed():
    lightning_client = mock.MagicMock()
    lightning_client.projects_service_get_project = mock.MagicMock(return_value=V1Project(id="project_id"))
    project = _get_project(lightning_client, project_id="project_id")
    assert project.project_id == "project_id"
    lightning_client.projects_service_get_project.assert_called_once_with("project_id")


def test_is_running_cloud():
    """We can determine if Lightning is running in the cloud."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not is_running_in_cloud()

    with mock.patch.dict(os.environ, {"LAI_RUNNING_IN_CLOUD": "0"}, clear=True):
        assert not is_running_in_cloud()

    # in the cloud, LIGHTNING_APP_STATE_URL is defined
    with mock.patch.dict(os.environ, {"LIGHTNING_APP_STATE_URL": "defined"}, clear=True):
        assert is_running_in_cloud()

    # LAI_RUNNING_IN_CLOUD is used to fake the value of `is_running_in_cloud` when loading the app for --cloud
    with mock.patch.dict(os.environ, {"LAI_RUNNING_IN_CLOUD": "1"}):
        assert is_running_in_cloud()
