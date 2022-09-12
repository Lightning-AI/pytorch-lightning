from unittest import mock
from unittest.mock import MagicMock

import pytest as pytest
from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1LightningappInstanceSpec,
    V1LightningappInstanceState,
    V1LightningappInstanceStatus,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)
from rich.text import Text

from lightning_app.cli.cmd_apps import _AppList, _AppManager


@pytest.mark.parametrize(
    "current_state,desired_state,expected",
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
@mock.patch("lightning_app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning_app.utilities.network.LightningClient.projects_service_list_memberships")
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
        mock.call(project_id="default-project", limit=100),
        mock.call(project_id="default-project", page_token="page-2", limit=100),
    ]


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning_app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_all_apps(list_memberships: mock.MagicMock, list_instances: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_instances.return_value = V1ListLightningappInstancesResponse(lightningapps=[])

    cluster_manager = _AppManager()
    cluster_manager.list()

    list_memberships.assert_called_once()
    list_instances.assert_called_once_with(project_id="default-project", limit=100)


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning_app.utilities.network.LightningClient.projects_service_list_memberships")
def test_list_apps_on_cluster(list_memberships: mock.MagicMock, list_instances: mock.MagicMock):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_instances.return_value = V1ListLightningappInstancesResponse(lightningapps=[])

    cluster_manager = _AppManager()
    cluster_manager.list(cluster_id="12345")

    list_memberships.assert_called_once()
    list_instances.assert_called_once_with(project_id="default-project", cluster_id="12345", limit=100)
