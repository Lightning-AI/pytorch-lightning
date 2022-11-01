from unittest import mock
from unittest.mock import call, MagicMock

import click
import pytest
from lightning_cloud.openapi import (
    Externalv1Cluster,
    V1AWSClusterDriverSpec,
    V1ClusterDriver,
    V1ClusterPerformanceProfile,
    V1ClusterSpec,
    V1ClusterState,
    V1ClusterStatus,
    V1ClusterType,
    V1CreateClusterRequest,
    V1KubernetesClusterDriver,
    V1ListClustersResponse,
)

from lightning_app.cli import cmd_clusters
from lightning_app.cli.cmd_clusters import AWSClusterManager


class FakeLightningClient:
    def __init__(self, list_responses=[], consume=True):
        self.list_responses = list_responses
        self.list_call_count = 0
        self.consume = consume

    def cluster_service_list_clusters(self, phase_not_in=None):
        self.list_call_count = self.list_call_count + 1
        if self.consume:
            return self.list_responses.pop(0)
        return self.list_responses[0]


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.cluster_service_create_cluster")
def test_create_cluster(api: mock.MagicMock):
    cluster_manager = AWSClusterManager()
    cluster_manager.create(
        cluster_name="test-7",
        external_id="dummy",
        role_arn="arn:aws:iam::1234567890:role/lai-byoc",
        region="us-west-2",
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
@mock.patch("lightning_app.utilities.network.LightningClient.cluster_service_list_clusters")
def test_list_clusters(api: mock.MagicMock):
    cluster_manager = AWSClusterManager()
    cluster_manager.list()

    api.assert_called_once_with(phase_not_in=[V1ClusterState.DELETED])


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.cluster_service_delete_cluster")
def test_delete_cluster(api: mock.MagicMock):
    cluster_manager = AWSClusterManager()
    cluster_manager.delete(cluster_id="test-7")

    api.assert_called_once_with(id="test-7", force=False)


class Test_check_cluster_name_is_valid:
    @pytest.mark.parametrize("name", ["test-7", "0wildgoat"])
    def test_valid(self, name):
        assert cmd_clusters._check_cluster_name_is_valid(None, None, name)

    @pytest.mark.parametrize(
        "name", ["(&%)!@#", "1234567890123456789012345678901234567890123456789012345678901234567890"]
    )
    def test_invalid(self, name):
        with pytest.raises(click.ClickException) as e:
            cmd_clusters._check_cluster_name_is_valid(None, None, name)
            assert "cluster name doesn't match regex pattern" in str(e.value)


class Test_wait_for_cluster_state:
    # TODO(rra) add tests for pagination

    @pytest.mark.parametrize("target_state", [V1ClusterState.RUNNING, V1ClusterState.DELETED])
    @pytest.mark.parametrize(
        "previous_state", [V1ClusterState.QUEUED, V1ClusterState.PENDING, V1ClusterState.UNSPECIFIED]
    )
    def test_happy_path(self, target_state, previous_state):
        client = FakeLightningClient(
            list_responses=[
                V1ListClustersResponse(
                    clusters=[Externalv1Cluster(id="test-cluster", status=V1ClusterStatus(phase=state))]
                )
                for state in [previous_state, target_state]
            ]
        )
        cmd_clusters._wait_for_cluster_state(client, "test-cluster", target_state, poll_duration=0.1)
        assert client.list_call_count == 2

    @pytest.mark.parametrize("target_state", [V1ClusterState.RUNNING, V1ClusterState.DELETED])
    def test_times_out(self, target_state):
        client = FakeLightningClient(
            list_responses=[
                V1ListClustersResponse(
                    clusters=[
                        Externalv1Cluster(id="test-cluster", status=V1ClusterStatus(phase=V1ClusterState.UNSPECIFIED))
                    ]
                )
            ],
            consume=False,
        )
        with pytest.raises(click.ClickException) as e:
            cmd_clusters._wait_for_cluster_state(client, "test-cluster", target_state, timeout=0.4, poll_duration=0.2)
            assert "Max wait time elapsed" in str(e.value)

    @mock.patch("click.echo")
    def test_echo_state_change_on_desired_running(self, echo: MagicMock):
        client = FakeLightningClient(
            list_responses=[
                V1ListClustersResponse(
                    clusters=[
                        Externalv1Cluster(
                            id="test-cluster",
                            status=V1ClusterStatus(
                                phase=state,
                                reason=reason,
                            ),
                        ),
                    ]
                )
                for state, reason in [
                    (V1ClusterState.QUEUED, ""),
                    (V1ClusterState.PENDING, ""),
                    (V1ClusterState.PENDING, ""),
                    (V1ClusterState.PENDING, ""),
                    (V1ClusterState.FAILED, "some error"),
                    (V1ClusterState.PENDING, "retrying failure"),
                    (V1ClusterState.RUNNING, ""),
                ]
            ],
        )

        cmd_clusters._wait_for_cluster_state(
            client,
            "test-cluster",
            target_state=V1ClusterState.RUNNING,
            timeout=0.6,
            poll_duration=0.1,
        )

        assert client.list_call_count == 7
        assert echo.call_count == 7
        echo.assert_has_calls(
            [
                call("Cluster test-cluster is now queued"),
                call("Cluster test-cluster is now pending"),
                call("Cluster test-cluster is now pending"),
                call("Cluster test-cluster is now pending"),
                call(
                    "\n".join(
                        [
                            "Cluster test-cluster is now failed with the following reason: some error",
                            "We are automatically retrying cluster creation.",
                            "In case you want to delete this cluster:",
                            "1. Stop this command",
                            "2. Run `lightning delete cluster test-cluster",
                            "WARNING: Any non-deleted cluster can consume cloud resources and incur cost to you.",
                        ]
                    )
                ),
                call("Cluster test-cluster is now pending with the following reason: retrying failure"),
                call(
                    "\n".join(
                        [
                            "Cluster test-cluster is now running and ready to use.",
                            "To launch an app on this cluster use "
                            "`lightning run app app.py --cloud --cluster-id test-cluster`",
                        ]
                    )
                ),
            ]
        )

    @mock.patch("click.echo")
    def test_echo_state_change_on_desired_deleted(self, echo: MagicMock):
        client = FakeLightningClient(
            list_responses=[
                V1ListClustersResponse(
                    clusters=[
                        Externalv1Cluster(
                            id="test-cluster",
                            status=V1ClusterStatus(
                                phase=state,
                            ),
                        ),
                    ]
                )
                for state in [
                    V1ClusterState.RUNNING,
                    V1ClusterState.RUNNING,
                    V1ClusterState.RUNNING,
                    V1ClusterState.DELETED,
                ]
            ],
        )

        cmd_clusters._wait_for_cluster_state(
            client,
            "test-cluster",
            target_state=V1ClusterState.DELETED,
            timeout=0.4,
            poll_duration=0.1,
        )

        assert client.list_call_count == 4
        assert echo.call_count == 4
        echo.assert_has_calls(
            [
                call("Cluster test-cluster is terminating"),
                call("Cluster test-cluster is terminating"),
                call("Cluster test-cluster is terminating"),
                call("Cluster test-cluster has been deleted."),
            ]
        )
