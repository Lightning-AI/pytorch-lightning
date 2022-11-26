import json
import re
import time
from datetime import datetime
from textwrap import dedent
from typing import Any, List

import click
from lightning_cloud.openapi import (
    Externalv1Cluster,
    V1AWSClusterDriverSpec,
    V1ClusterDriver,
    V1ClusterPerformanceProfile,
    V1ClusterSpec,
    V1ClusterState,
    V1ClusterType,
    V1CreateClusterRequest,
    V1KubernetesClusterDriver,
)
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lightning_app.cli.core import Formatable
from lightning_app.utilities.network import LightningClient
from lightning_app.utilities.openapi import create_openapi_object, string2dict

CLUSTER_STATE_CHECKING_TIMEOUT = 60
MAX_CLUSTER_WAIT_TIME = 5400


class ClusterList(Formatable):
    def __init__(self, clusters: List[Externalv1Cluster]):
        self.clusters = clusters

    def as_json(self) -> str:
        return json.dumps(self.clusters)

    def as_table(self) -> Table:
        table = Table("id", "name", "type", "status", "created", show_header=True, header_style="bold green")
        phases = {
            V1ClusterState.QUEUED: Text("queued", style="bold yellow"),
            V1ClusterState.PENDING: Text("pending", style="bold yellow"),
            V1ClusterState.RUNNING: Text("running", style="bold green"),
            V1ClusterState.FAILED: Text("error", style="bold red"),
            V1ClusterState.DELETED: Text("deleted", style="bold red"),
        }

        cluster_type_lookup = {
            V1ClusterType.BYOC: Text("byoc", style="bold yellow"),
            V1ClusterType.GLOBAL: Text("lightning-cloud", style="bold green"),
        }
        for cluster in self.clusters:
            status = phases[cluster.status.phase]
            if cluster.spec.desired_state == V1ClusterState.DELETED and cluster.status.phase != V1ClusterState.DELETED:
                status = Text("terminating", style="bold red")

            # this guard is necessary only until 0.3.93 releases which includes the `created_at`
            # field to the external API
            created_at = datetime.now()
            if hasattr(cluster, "created_at"):
                created_at = cluster.created_at

            table.add_row(
                cluster.id,
                cluster.name,
                cluster_type_lookup.get(cluster.spec.cluster_type, Text("unknown", style="red")),
                status,
                created_at.strftime("%Y-%m-%d") if created_at else "",
            )
        return table


class AWSClusterManager:
    """AWSClusterManager implements API calls specific to Lightning AI BYOC compute clusters when the AWS provider
    is selected as the backend compute."""

    def __init__(self) -> None:
        self.api_client = LightningClient()

    def create(
        self,
        cost_savings: bool = False,
        cluster_name: str = None,
        role_arn: str = None,
        region: str = "us-east-1",
        external_id: str = None,
        edit_before_creation: bool = False,
        wait: bool = False,
    ) -> None:
        """request Lightning AI BYOC compute cluster creation.

        Args:
            cost_savings: Specifies if the cluster uses cost savings mode
            cluster_name: The name of the cluster to be created
            role_arn: AWS IAM Role ARN used to provision resources
            region: AWS region containing compute resources
            external_id: AWS IAM Role external ID
            edit_before_creation: Enables interactive editing of requests before submitting it to Lightning AI.
            wait: Waits for the cluster to be in a RUNNING state. Only use this for debugging.
        """
        performance_profile = V1ClusterPerformanceProfile.DEFAULT
        if cost_savings:
            """In cost saving mode the number of compute nodes is reduced to one, reducing the cost for clusters
            with low utilization."""
            performance_profile = V1ClusterPerformanceProfile.COST_SAVING

        body = V1CreateClusterRequest(
            name=cluster_name,
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.BYOC,
                performance_profile=performance_profile,
                driver=V1ClusterDriver(
                    kubernetes=V1KubernetesClusterDriver(
                        aws=V1AWSClusterDriverSpec(
                            region=region,
                            role_arn=role_arn,
                            external_id=external_id,
                        )
                    )
                ),
            ),
        )
        new_body = body
        if edit_before_creation:
            after = click.edit(json.dumps(body.to_dict(), indent=4))
            if after is not None:
                new_body = create_openapi_object(string2dict(after), body)
            if new_body == body:
                click.echo("cluster unchanged")

        resp = self.api_client.cluster_service_create_cluster(body=new_body)
        if wait:
            _wait_for_cluster_state(self.api_client, resp.id, V1ClusterState.RUNNING)

        click.echo(
            dedent(
                f"""\
            {resp.id} is now being created... This can take up to an hour.

            To view the status of your clusters use:
                `lightning list clusters`

            To view cluster logs use:
                `lightning show cluster logs {resp.id}`
                    """
            )
        )

    def get_clusters(self) -> ClusterList:
        resp = self.api_client.cluster_service_list_clusters(phase_not_in=[V1ClusterState.DELETED])
        return ClusterList(resp.clusters)

    def list(self) -> None:
        clusters = self.get_clusters()
        console = Console()
        console.print(clusters.as_table())

    def delete(self, cluster_id: str, force: bool = False, wait: bool = False) -> None:
        if force:
            click.echo(
                """
            Deletes a BYOC cluster. Lightning AI removes cluster artifacts and any resources running on the cluster.\n
            WARNING: Deleting a cluster does not clean up any resources managed by Lightning AI.\n
            Check your cloud provider to verify that existing cloud resources are deleted.
            """
            )
            click.confirm("Do you want to continue?", abort=True)

        self.api_client.cluster_service_delete_cluster(id=cluster_id, force=force)
        click.echo("Cluster deletion triggered successfully")

        if wait:
            _wait_for_cluster_state(self.api_client, cluster_id, V1ClusterState.DELETED)


def _wait_for_cluster_state(
    api_client: LightningClient,
    cluster_id: str,
    target_state: V1ClusterState,
    max_wait_time: int = MAX_CLUSTER_WAIT_TIME,
    check_timeout: int = CLUSTER_STATE_CHECKING_TIMEOUT,
) -> None:
    """_wait_for_cluster_state waits until the provided cluster has reached a desired state, or failed.

    Args:
        api_client: LightningClient used for polling
        cluster_id: Specifies the cluster to wait for
        target_state: Specifies the desired state the target cluster needs to meet
        max_wait_time: Maximum duration to wait (in seconds)
        check_timeout: duration between polling for the cluster state (in seconds)
    """
    start = time.time()
    elapsed = 0
    while elapsed < max_wait_time:
        cluster_resp = api_client.cluster_service_list_clusters()
        new_cluster = None
        for clust in cluster_resp.clusters:
            if clust.id == cluster_id:
                new_cluster = clust
                break
        if new_cluster is not None:
            if new_cluster.status.phase == target_state:
                break
            elif new_cluster.status.phase == V1ClusterState.FAILED:
                raise click.ClickException(f"Cluster {cluster_id} is in failed state.")
            time.sleep(check_timeout)
        elapsed = int(time.time() - start)
    else:
        raise click.ClickException("Max wait time elapsed")


def _check_cluster_name_is_valid(_ctx: Any, _param: Any, value: str) -> str:
    pattern = r"^(?!-)[a-z0-9-]{1,63}(?<!-)$"
    if not re.match(pattern, value):
        raise click.ClickException(
            """The cluster name is invalid.
            Cluster names can only contain lowercase letters, numbers, and periodic hyphens ( - ).
            Provide a cluster name using valid characters and try again."""
        )
    return value
