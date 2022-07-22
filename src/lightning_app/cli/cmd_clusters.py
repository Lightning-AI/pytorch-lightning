import json
from datetime import datetime
from dataclasses import dataclass

from lightning_cloud.openapi import V1ClusterPerformanceProfile, V1CreateClusterRequest, V1ClusterSpec, V1ClusterDriver, \
    V1KubernetesClusterDriver, V1AWSClusterDriverSpec, V1InstanceSpec
from rich.console import Console
from rich.table import Table
from rich.text import Text
import arrow
import click
import time
import re

from lightning_cloud.openapi.models import (
    V1ClusterState,
    Externalv1Cluster,
    V1ClusterType,
)

from lightning_app.cli.core import Formatable
from lightning_app.utilities.network import LightningClient
from lightning_app.utilities.openapi import create_openapi_object, string2dict

CLUSTER_STATE_CHECKING_TIMEOUT = 60
MAX_CLUSTER_WAIT_TIME = 5400

class AWSClusterManager:
    def __init__(self):
        self.api_client = LightningClient()

    def create(self, cost_savings=None, cluster_name=None, role_arn=None, region=None, external_id=None,
               instance_types=None, edit_before_creation=None, wait=None):
        performance_profile = V1ClusterPerformanceProfile.DEFAULT
        if cost_savings:
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
                            instance_types=[V1InstanceSpec(name=x) for x in instance_types.split(",")]
                        )
                    )
                )
            )
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

        click.echo(resp.to_str())

    def list(self):
        resp = self.api_client.cluster_service_list_clusters(phase_not_in=[V1ClusterState.DELETED])
        console = Console()
        console.print(ClusterList(resp.clusters).as_table())

    def delete(self, cluster_id=None, force=None, wait=None):
        if force:
            click.echo(
                "Force deleting cluster. This will cause grid to forget "
                "about the cluster and any experiments, sessions, datastores, "
                "tensorboards and other resources running on it.\n"
                "WARNING: this will not clean up any resources managed by grid\n"
                "Check your cloud provider that any existing cloud resources are deleted"
                )
            click.confirm('Do you want to continue?', abort=True)

        self.api_client.cluster_service_delete_cluster(id=cluster_id, force=force)
        click.echo("Cluster deletion triggered successfully")

        if wait:
            _wait_for_cluster_state(self.api_client, cluster_id, V1ClusterState.DELETED)

class ClusterList(Formatable):
    def __init__(self, clusters: [Externalv1Cluster]):
        self.clusters = clusters

    def as_json(self) -> str:
        return json.dumps(self.clusters)

    def as_table(self) -> Table:
        table = Table("id", "name", "type", "status", "created", show_header=True, header_style="bold green")
        phases = {
            V1ClusterState.QUEUED: Text("queued", style="bold yellow"),
            V1ClusterState.PENDING: Text("pending", style="bold yellow"),
            V1ClusterState.RUNNING: Text("running", style="bold green"),
            V1ClusterState.FAILED: Text("failed", style="bold red"),
            V1ClusterState.DELETED: Text("deleted", style="bold red"),
        }

        cluster_type_lookup = {
            V1ClusterType.BYOC: Text("byoc", style="bold yellow"),
            V1ClusterType.GLOBAL: Text("grid-cloud", style="bold green"),
        }
        for cluster in self.clusters:
            cluster: Externalv1Cluster
            status = phases[cluster.status.phase]
            if cluster.spec.desired_state == V1ClusterState.DELETED \
                    and cluster.status.phase != V1ClusterState.DELETED:
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
                arrow.get(created_at).humanize() if created_at else "",
            )
        return table


def _wait_for_cluster_state(
        api_client,
        cluster_id: str,
        target_state: V1ClusterState,
        max_wait_time=MAX_CLUSTER_WAIT_TIME,
        check_timeout=CLUSTER_STATE_CHECKING_TIMEOUT,
):
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
                raise click.ClickException(f"cluster {cluster_id} is in failed state")
            time.sleep(check_timeout)
        elapsed = time.time() - start
    else:
        raise click.ClickException(f"Max wait time elapsed")


def _check_cluster_name_is_valid(_ctx, _param, value):
    pattern = r"^(?!-)[a-z0-9-]{1,63}(?<!-)$"
    if not re.match(pattern, value):
        raise click.ClickException(
            f"cluster name doesn't match regex pattern {pattern}\nIn simple words, use lowercase letters, numbers, and occasional -"
        )
    return value


default_instance_types = [
    "g2.8xlarge",
    "g3.16xlarge",
    "g3.4xlarge",
    "g3.8xlarge",
    "g3s.xlarge",
    "g4dn.12xlarge",
    "g4dn.16xlarge",
    "g4dn.2xlarge",
    "g4dn.4xlarge",
    "g4dn.8xlarge",
    "g4dn.metal",
    "g4dn.xlarge",
    "p2.16xlarge",
    "p2.8xlarge",
    "p2.xlarge",
    "p3.16xlarge",
    "p3.2xlarge",
    "p3.8xlarge",
    "p3dn.24xlarge",
    # "p4d.24xlarge",  # currently not supported
    "t2.large",
    "t2.medium",
    "t2.xlarge",
    "t2.2xlarge",
    "t3.large",
    "t3.medium",
    "t3.xlarge",
    "t3.2xlarge",
]
