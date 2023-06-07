from unittest import mock
from unittest.mock import MagicMock

import click
import pytest
from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1AWSClusterDriverSpec,
    V1ClusterDriver,
    V1ClusterPerformanceProfile,
    V1ClusterSpec,
    V1ClusterState,
    V1ClusterStatus,
    V1ClusterType,
    V1CreateClusterRequest,
    V1GetClusterResponse,
    V1KubernetesClusterDriver,
    V1LightningappInstanceState,
    V1LightningappInstanceStatus,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)

from lightning.app.cli import cmd_clusters
from lightning.app.cli.cmd_clusters import AWSClusterManager


@pytest.fixture(params=[True, False])
def async_or_interrupt(request, monkeypatch):
    # Simulate hitting ctrl-c immediately while waiting for cluster to create
    if not request.param:
        monkeypatch.setattr(cmd_clusters, "_wait_for_cluster_state", mock.MagicMock(side_effect=KeyboardInterrupt))
    return request.param


@pytest.fixture()
def spec():
    return V1ClusterSpec(
        driver=V1ClusterDriver(
            kubernetes=V1KubernetesClusterDriver(
                aws=V1AWSClusterDriverSpec(
                    bucket_name="test-bucket",
                ),
            ),
        ),
    )


class FakeLightningClient:
    def __init__(self, get_responses=[], consume=True):
        self.get_responses = get_responses
        self.get_call_count = 0
        self.consume = consume

    def cluster_service_get_cluster(self, id: str):
        self.get_call_count = self.get_call_count + 1
        if self.consume:
            return self.get_responses.pop(0)
        return self.get_responses[0]


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_create_cluster")
def test_create_cluster_api(api: mock.MagicMock, async_or_interrupt):
    cluster_manager = AWSClusterManager()
    cluster_manager.create(
        cluster_id="test-7",
        external_id="dummy",
        role_arn="arn:aws:iam::1234567890:role/lai-byoc",
        region="us-west-2",
        do_async=async_or_interrupt,
    )

    api.assert_called_once_with(
        body=V1CreateClusterRequest(
            name="test-7",
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.BYOC,
                performance_profile=V1ClusterPerformanceProfile.DEFAULT,
                driver=V1ClusterDriver(
                    kubernetes=V1KubernetesClusterDriver(
                        aws=V1AWSClusterDriverSpec(
                            region="us-west-2",
                            role_arn="arn:aws:iam::1234567890:role/lai-byoc",
                            external_id="dummy",
                        )
                    )
                ),
            ),
        )
    )


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_list_clusters")
def test_list_clusters(api: mock.MagicMock):
    cluster_manager = AWSClusterManager()
    cluster_manager.list()

    api.assert_called_once_with(phase_not_in=[V1ClusterState.DELETED])


@pytest.fixture()
def fixture_list_instances_empty():
    return V1ListLightningappInstancesResponse([])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
@mock.patch("lightning.app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_delete_cluster")
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_get_cluster")
def test_delete_cluster_api(
    api_get: mock.MagicMock,
    api_delete: mock.MagicMock,
    api_list_instances: mock.MagicMock,
    api_list_memberships: mock.MagicMock,
    async_or_interrupt,
    spec,
    fixture_list_instances_empty,
):
    api_list_memberships.return_value = V1ListMembershipsResponse([V1Membership(project_id="test-project")])
    api_list_instances.return_value = fixture_list_instances_empty
    api_get.return_value = V1GetClusterResponse(spec=spec)
    cluster_manager = AWSClusterManager()
    cluster_manager.delete(cluster_id="test-7", do_async=async_or_interrupt)

    api_delete.assert_called_once_with(id="test-7", force=False)


@mock.patch("click.confirm")
@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
@mock.patch("lightning.app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_delete_cluster")
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_get_cluster")
def test_delete_cluster_with_stopped_apps(
    api_get: mock.MagicMock,
    api_delete: mock.MagicMock,
    api_list_instances: mock.MagicMock,
    api_list_memberships: mock.MagicMock,
    click_mock: mock.MagicMock,
    spec,
):
    api_list_memberships.return_value = V1ListMembershipsResponse([V1Membership(project_id="test-project")])
    api_list_instances.side_effect = [
        # when querying for running apps
        V1ListLightningappInstancesResponse([]),
        # when querying for stopped apps
        V1ListLightningappInstancesResponse(
            lightningapps=[
                Externalv1LightningappInstance(
                    status=V1LightningappInstanceStatus(
                        phase=V1LightningappInstanceState.STOPPED,
                    )
                )
            ]
        ),
    ]
    api_get.return_value = V1GetClusterResponse(spec=spec)
    cluster_manager = AWSClusterManager()

    cluster_manager.delete(cluster_id="test-7", do_async=True)
    api_delete.assert_called_once_with(id="test-7", force=False)
    click_mock.assert_called_once_with("Are you sure you want to continue?", abort=True)


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.utilities.network.LightningClient.projects_service_list_memberships")
@mock.patch("lightning.app.utilities.network.LightningClient.lightningapp_instance_service_list_lightningapp_instances")
@mock.patch("lightning.app.utilities.network.LightningClient.cluster_service_get_cluster")
def test_delete_cluster_with_running_apps(
    api_get: mock.MagicMock,
    api_list_instances: mock.MagicMock,
    api_list_memberships: mock.MagicMock,
    spec,
):
    api_list_memberships.return_value = V1ListMembershipsResponse([V1Membership(project_id="test-project")])
    api_list_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[
            Externalv1LightningappInstance(
                status=V1LightningappInstanceStatus(
                    phase=V1LightningappInstanceState.RUNNING,
                )
            )
        ]
    )
    api_get.return_value = V1GetClusterResponse(spec=spec)
    cluster_manager = AWSClusterManager()

    with pytest.raises(click.ClickException) as exception:
        cluster_manager.delete(cluster_id="test-7")
    exception.match("apps running")


class Test_check_cluster_id_is_valid:
    @pytest.mark.parametrize("name", ["test-7", "0wildgoat"])
    def test_valid(self, name):
        assert cmd_clusters._check_cluster_id_is_valid(None, None, name)

    @pytest.mark.parametrize(
        "name", ["(&%)!@#", "1234567890123456789012345678901234567890123456789012345678901234567890"]
    )
    def test_invalid(self, name):
        with pytest.raises(click.ClickException) as e:
            cmd_clusters._check_cluster_id_is_valid(None, None, name)
            assert "cluster name doesn't match regex pattern" in str(e.value)


class Test_wait_for_cluster_state:
    # TODO(rra) add tests for pagination

    @pytest.mark.parametrize("target_state", [V1ClusterState.RUNNING, V1ClusterState.DELETED])
    @pytest.mark.parametrize(
        "previous_state", [V1ClusterState.QUEUED, V1ClusterState.PENDING, V1ClusterState.UNSPECIFIED]
    )
    def test_happy_path(self, target_state, previous_state, spec):
        client = FakeLightningClient(
            get_responses=[
                V1GetClusterResponse(
                    id="test-cluster",
                    status=V1ClusterStatus(phase=state),
                    spec=spec
                )
                for state in [previous_state, target_state]
            ]
        )
        cmd_clusters._wait_for_cluster_state(client, "test-cluster", target_state, poll_duration_seconds=0.1)
        assert client.get_call_count == 2

    @pytest.mark.parametrize("target_state", [V1ClusterState.RUNNING, V1ClusterState.DELETED])
    def test_times_out(self, target_state, spec):
        client = FakeLightningClient(
            get_responses=[
                V1GetClusterResponse(
                    id="test-cluster",
                    status=V1ClusterStatus(phase=V1ClusterState.UNSPECIFIED),
                    spec=spec
                )
            ],
            consume=False,
        )
        with pytest.raises(click.ClickException) as e:
            cmd_clusters._wait_for_cluster_state(
                client, "test-cluster", target_state, timeout_seconds=0.1, poll_duration_seconds=0.1
            )

        if target_state == V1ClusterState.DELETED:
            expected_state = "deleted"
        if target_state == V1ClusterState.RUNNING:
            expected_state = "running"

        assert e.match(f"The cluster has not entered the {expected_state} state")
