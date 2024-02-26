from unittest import mock
from unittest.mock import MagicMock

import pytest as pytest
from lightning.app.cli.cmd_apps import _AppList, _AppManager
from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1LightningappInstanceSpec,
    V1LightningappInstanceState,
    V1LightningappInstanceStatus,
    V1LightningworkState,
    V1ListLightningappInstancesResponse,
    V1ListLightningworkResponse,
    V1ListMembershipsResponse,
    V1Membership,
)
from rich.text import Text


@pytest.mark.parametrize(
    ("current_state", "desired_state", "expected"),
    [
        (
            V1LightningappInstanceStatus(phase=V1LightningappInstanceState.RUNNING),
            V1LightningappInstanceState.DELETED,
            Text("terminating"),
        ),
        (
            V1LightningappInstanceStatus(phase=V1LightningappInstanceState.STOPPED),
            V1LightningappInstanceState.RUNNING,
            Text("restarting"),
        ),
        (
            V1LightningappInstanceStatus(phase=V1LightningappInstanceState.PENDING),
            V1LightningappInstanceState.RUNNING,
            Text("restarting"),
        ),
        (
            V1LightningappInstanceStatus(phase=V1LightningappInstanceState.UNSPECIFIED, start_timestamp=None),
            V1LightningappInstanceState.RUNNING,
            Text("not yet started"),
        ),
    ],
)
def test_state_transitions(current_state, desired_state, expected):
    actual = _AppList._textualize_state_transitions(current_state=current_state, desired_state=desired_state)
    assert actual == expected


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_all_apps_paginated(list_memberships: mock.MagicMock, list_instances: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_instances.side_effect = [
        V1ListLightningappInstancesResponse(
            lightningapps=[
                Externalv1LightningappInstance(
                    name="test1",
                    spec=V1LightningappInstanceSpec(desired_state=V1LightningappInstanceState.RUNNING),
                    status=V1LightningappInstanceStatus(phase=V1LightningappInstanceState.RUNNING),
                )
            ],
            next_page_token="page-2",
        ),
        V1ListLightningappInstancesResponse(
            lightningapps=[
                Externalv1LightningappInstance(
                    name="test2",
                    spec=V1LightningappInstanceSpec(desired_state=V1LightningappInstanceState.STOPPED),
                    status=V1LightningappInstanceStatus(phase=V1LightningappInstanceState.RUNNING),
                )
            ],
        ),
    ]

    cluster_manager = _AppManager()
    cluster_manager.list()

    list_memberships.assert_called_once()
    assert list_instances.mock_calls == [
        mock.call(project_id="default-project", limit=100, phase_in=[]),
        mock.call(project_id="default-project", page_token="page-2", limit=100, phase_in=[]),
    ]


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_all_apps(list_memberships: mock.MagicMock, list_instances: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_instances.return_value = V1ListLightningappInstancesResponse(lightningapps=[])

    cluster_manager = _AppManager()
    cluster_manager.list()

    list_memberships.assert_called_once()
    list_instances.assert_called_once_with(project_id="default-project", limit=100, phase_in=[])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.lightningwork_service_list_lightningwork")
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_components(list_memberships: mock.MagicMock, list_components: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_components.return_value = V1ListLightningworkResponse(lightningworks=[])

    cluster_manager = _AppManager()
    cluster_manager.list_components(app_id="cheese")

    list_memberships.assert_called_once()
    list_components.assert_called_once_with(project_id="default-project", app_id="cheese", phase_in=[])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.lightningwork_service_list_lightningwork")
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_components_with_phase(list_memberships: mock.MagicMock, list_components: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_components.return_value = V1ListLightningworkResponse(lightningworks=[])

    cluster_manager = _AppManager()
    cluster_manager.list_components(app_id="cheese", phase_in=[V1LightningworkState.RUNNING])

    list_memberships.assert_called_once()
    list_components.assert_called_once_with(
        project_id="default-project", app_id="cheese", phase_in=[V1LightningworkState.RUNNING]
    )


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_apps_on_cluster(list_memberships: mock.MagicMock, list_instances: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_instances.return_value = V1ListLightningappInstancesResponse(lightningapps=[])

    cluster_manager = _AppManager()
    cluster_manager.list()

    list_memberships.assert_called_once()
    list_instances.assert_called_once_with(project_id="default-project", limit=100, phase_in=[])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch(
    "lightning.app.utilities.network.LightningClient.lightningapp_instance_service_delete_lightningapp_instance"
)
@mock.patch("lightning.app.cli.cmd_apps._get_project")
def test_delete_app_on_cluster(get_project_mock: mock.MagicMock, delete_app_mock: mock.MagicMock):
    get_project_mock.return_value = V1Membership(project_id="default-project")

    cluster_manager = _AppManager()
    cluster_manager.delete(app_id="12345")

    delete_app_mock.assert_called()
    delete_app_mock.assert_called_once_with(project_id="default-project", id="12345")
