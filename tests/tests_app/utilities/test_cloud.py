import os
from unittest import mock

from lightning.app.utilities.cloud import _get_project, is_running_in_cloud
from lightning_cloud.openapi.models import V1ListMembershipsResponse, V1Membership
from lightning.app.utilities.network import LightningClient


@mock.patch("lightning.app.core.constants.LIGHTNING_CLOUD_ORGANIZATION_ID", "organization_id")
def test_get_project_picks_up_organization_id():
    """Uses organization_id from `LIGHTNING_CLOUD_ORGANIZATION_ID` config var if none passed"""
    lightning_client = mock.MagicMock()
    lightning_client.projects_service_list_memberships = mock.MagicMock(
        return_value=V1ListMembershipsResponse(memberships=[V1Membership(project_id="project_id")]),
    )
    _get_project(lightning_client)
    lightning_client.projects_service_list_memberships.assert_called_once_with(organization_id="organization_id")


def test_get_project_doesnt_pass_organization_id_if_its_not_set():
    lightning_client = mock.MagicMock()
    lightning_client.projects_service_list_memberships = mock.MagicMock(
        return_value=V1ListMembershipsResponse(memberships=[V1Membership(project_id="project_id")]),
    )
    _get_project(lightning_client)
    lightning_client.projects_service_list_memberships.assert_called_once_with()


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
