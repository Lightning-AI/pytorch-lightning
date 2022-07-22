import pytest
import click
from lightning_cloud.openapi.models import (
    V1ClusterState,
    V1ListClustersResponse,
    Externalv1Cluster,
    V1ClusterStatus,
)
from lightning_app.cli import cmd_clusters


class FakeLightningClient:
    def __init__(self, list_responses=[], consume=True):
        self.list_responses = list_responses
        self.list_call_count = 0
        self.consume = consume

    def cluster_service_list_clusters(self, phase_not_in=None):
        self.list_call_count = self.list_call_count + 1
        if self.consume:
            return self.list_responses.pop()
        return self.list_responses[0]


class Test_wait_for_cluster_state():
    # TODO(rra) add tests for pagination

    @pytest.mark.parametrize("target_state", [V1ClusterState.RUNNING, V1ClusterState.DELETED])
    @pytest.mark.parametrize("previous_state", [V1ClusterState.QUEUED, V1ClusterState.PENDING, V1ClusterState.UNSPECIFIED])
    def test_happy_path(self, target_state, previous_state):
        client = FakeLightningClient(list_responses=[V1ListClustersResponse(
            clusters=[Externalv1Cluster(
                id='test-cluster',
                status=V1ClusterStatus(
                    phase=state
                )
            )]
        ) for state in [previous_state, target_state]])
        cmd_clusters._wait_for_cluster_state(client, "test-cluster", target_state, check_timeout=0.1)
        assert client.list_call_count == 1

    @pytest.mark.parametrize("target_state", [V1ClusterState.RUNNING, V1ClusterState.DELETED])
    def test_times_out(self, target_state):
        client = FakeLightningClient(list_responses=[V1ListClustersResponse(
            clusters=[Externalv1Cluster(
                id='test-cluster',
                status=V1ClusterStatus(
                    phase=V1ClusterState.UNSPECIFIED
                )
            )]
        )], consume=False)
        with pytest.raises(click.ClickException) as e:
            cmd_clusters._wait_for_cluster_state(
                client,
                "test-cluster",
                target_state,
                max_wait_time=0.4,
                check_timeout=0.2)
            assert "Max wait time elapsed" in str(e.value)

