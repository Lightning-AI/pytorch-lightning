# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import time
from datetime import datetime
from textwrap import dedent
from typing import Any, List, Union

import click
import lightning_cloud
from lightning_cloud.openapi import (
    Externalv1Cluster,
    Externalv1LightningappInstance,
    V1AWSClusterDriverSpec,
    V1ClusterDriver,
    V1ClusterPerformanceProfile,
    V1ClusterSpec,
    V1ClusterState,
    V1ClusterType,
    V1CreateClusterRequest,
    V1GetClusterResponse,
    V1KubernetesClusterDriver,
    V1LightningappInstanceState,
    V1ListLightningappInstancesResponse,
    V1Membership,
)
from lightning_utilities.core.enums import StrEnum
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lightning.app.cli.core import Formatable
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient
from lightning.app.utilities.openapi import create_openapi_object, string2dict

MAX_CLUSTER_WAIT_TIME = 5400


class ClusterState(StrEnum):
    UNSPECIFIED = "unspecified"
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "error"
    DELETED = "deleted"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def from_api(cls, status: V1ClusterState) -> "ClusterState":
        parsed = str(status).lower().split("_", maxsplit=2)[-1]
        return cls.from_str(parsed)


class ClusterList(Formatable):
    def __init__(self, clusters: List[Externalv1Cluster]):
        self.clusters = clusters

    def as_json(self) -> str:
        return json.dumps(self.clusters)

    def as_table(self) -> Table:
        table = Table("id", "type", "status", "created", show_header=True, header_style="bold green")
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
                cluster_type_lookup.get(cluster.spec.cluster_type, Text("unknown", style="red")),
                status,
                created_at.strftime("%Y-%m-%d") if created_at else "",
            )
        return table


class AWSClusterManager:
    """AWSClusterManager implements API calls specific to Lightning AI BYOC compute clusters when the AWS provider
    is selected as the backend compute."""

    def __init__(self) -> None:
        self.api_client = LightningClient(retry=False)

    def create(
        self,
        cost_savings: bool = False,
        cluster_id: str = None,
        role_arn: str = None,
        region: str = "us-east-1",
        external_id: str = None,
        edit_before_creation: bool = False,
        do_async: bool = True,
    ) -> None:
        """request Lightning AI BYOC compute cluster creation.

        Args:
            cost_savings: Specifies if the cluster uses cost savings mode
            cluster_id: The name of the cluster to be created
            role_arn: AWS IAM Role ARN used to provision resources
            region: AWS region containing compute resources
            external_id: AWS IAM Role external ID
            edit_before_creation: Enables interactive editing of requests before submitting it to Lightning AI.
            do_async: Triggers cluster creation in the background and exits
        """
        performance_profile = V1ClusterPerformanceProfile.DEFAULT
        if cost_savings:
            """In cost saving mode the number of compute nodes is reduced to one, reducing the cost for clusters
            with low utilization."""
            performance_profile = V1ClusterPerformanceProfile.COST_SAVING

        body = V1CreateClusterRequest(
            name=cluster_id,
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
        click.echo(
            dedent(
                f"""\
            BYOC cluster creation triggered successfully!
            This can take up to an hour to complete.

            To view the status of your clusters use:
                lightning list clusters

            To view cluster logs use:
                lightning show cluster logs {cluster_id}

            To delete the cluster run:
                lightning delete cluster {cluster_id}
            """
            )
        )
        background_message = "\nCluster will be created in the background!"
        if do_async:
            click.echo(background_message)
        else:
            try:
                _wait_for_cluster_state(self.api_client, resp.id, V1ClusterState.RUNNING)
            except KeyboardInterrupt:
                click.echo(background_message)

    def list_clusters(self) -> List[Externalv1Cluster]:
        resp = self.api_client.cluster_service_list_clusters(phase_not_in=[V1ClusterState.DELETED])
        return resp.clusters

    def get_clusters(self) -> ClusterList:
        resp = self.api_client.cluster_service_list_clusters(phase_not_in=[V1ClusterState.DELETED])
        return ClusterList(resp.clusters)

    def list(self) -> None:
        clusters = self.get_clusters()
        console = Console()
        console.print(clusters.as_table())

    def delete(self, cluster_id: str, force: bool = False, do_async: bool = True) -> None:
        if force:
            click.echo(
                """
            Deletes a BYOC cluster. Lightning AI removes cluster artifacts and any resources running on the cluster.\n
            WARNING: Deleting a cluster does not clean up any resources managed by Lightning AI.\n
            Check your cloud provider to verify that existing cloud resources are deleted.
            """
            )
            click.confirm("Do you want to continue?", abort=True)

        else:
            apps = _list_apps(self.api_client, cluster_id=cluster_id, phase_in=[V1LightningappInstanceState.RUNNING])
            if apps:
                raise click.ClickException(
                    dedent(
                        """\
                        To delete the cluster, you must first delete the apps running on it.
                        Use the following commands to delete the apps, then delete the cluster again:

                        """
                    )
                    + "\n".join([f"\tlightning delete app {app.name} --cluster-id {cluster_id}" for app in apps])
                )

            if _list_apps(self.api_client, cluster_id=cluster_id, phase_not_in=[V1LightningappInstanceState.DELETED]):
                click.echo(
                    dedent(
                        """\
                        This cluster has stopped apps.
                        Deleting this cluster will delete those apps and their logs.

                        App artifacts aren't deleted and will still be available in the S3 bucket for the cluster.
                        """
                    )
                )
                click.confirm("Are you sure you want to continue?", abort=True)

        resp: V1GetClusterResponse = self.api_client.cluster_service_get_cluster(id=cluster_id)
        bucket_name = resp.spec.driver.kubernetes.aws.bucket_name

        self.api_client.cluster_service_delete_cluster(id=cluster_id, force=force)
        click.echo(
            dedent(
                f"""\
            Cluster deletion triggered successfully

            For safety purposes we will not delete anything in the S3 bucket associated with the cluster:
                {bucket_name}

            You may want to delete it manually using the AWS CLI:
                aws s3 rb --force s3://{bucket_name}
            """
            )
        )

        background_message = "\nCluster will be deleted in the background!"
        if do_async:
            click.echo(background_message)
        else:
            try:
                _wait_for_cluster_state(self.api_client, cluster_id, V1ClusterState.DELETED)
            except KeyboardInterrupt:
                click.echo(background_message)


def _list_apps(
    api_client: LightningClient,
    **filters: Any,
) -> List[Externalv1LightningappInstance]:
    """_list_apps is a thin wrapper around lightningapp_instance_service_list_lightningapp_instances.

    Args:
        api_client (LightningClient): Used for listing app instances
        **filters: keyword arguments passed to the list method

    Returns:
        List[Externalv1LightningappInstance]: List of apps matching the filters
    """
    project: V1Membership = _get_project(api_client)
    resp: V1ListLightningappInstancesResponse = api_client.lightningapp_instance_service_list_lightningapp_instances(
        project.project_id,
        **filters,
    )

    return resp.lightningapps


def _wait_for_cluster_state(
    api_client: LightningClient,
    cluster_id: str,
    target_state: V1ClusterState,
    timeout_seconds: int = MAX_CLUSTER_WAIT_TIME,
    poll_duration_seconds: int = 60,
) -> None:
    """_wait_for_cluster_state waits until the provided cluster has reached a desired state, or failed.

    Messages will be displayed to the user as the cluster changes state.
    We poll the API server for any changes

    Args:
        api_client: LightningClient used for polling
        cluster_id: Specifies the cluster to wait for
        target_state: Specifies the desired state the target cluster needs to meet
        timeout_seconds: Maximum duration to wait
        poll_duration_seconds: duration between polling for the cluster state
    """
    start = time.time()
    elapsed = 0

    click.echo(f"Waiting for cluster to be {ClusterState.from_api(target_state)}...")
    while elapsed < timeout_seconds:
        try:
            resp: V1GetClusterResponse = api_client.cluster_service_get_cluster(id=cluster_id)
            click.echo(_cluster_status_long(cluster=resp, desired_state=target_state, elapsed=elapsed))
            if resp.status.phase == target_state:
                break
            time.sleep(poll_duration_seconds)
            elapsed = int(time.time() - start)
        except lightning_cloud.openapi.rest.ApiException as e:
            if e.status == 404 and target_state == V1ClusterState.DELETED:
                return
            raise
    else:
        state_str = ClusterState.from_api(target_state)
        raise click.ClickException(
            dedent(
                f"""\
            The cluster has not entered the {state_str} state within {_format_elapsed_seconds(timeout_seconds)}.

            The cluster may eventually be {state_str} afterwards, please check its status using:
                lighting list clusters

            To view cluster logs use:
                lightning show cluster logs {cluster_id}

            Contact support@lightning.ai for additional help
            """
            )
        )


def _check_cluster_id_is_valid(_ctx: Any, _param: Any, value: str) -> str:
    pattern = r"^(?!-)[a-z0-9-]{1,63}(?<!-)$"
    if not re.match(pattern, value):
        raise click.ClickException(
            """The cluster name is invalid.
            Cluster names can only contain lowercase letters, numbers, and periodic hyphens ( - ).
            Provide a cluster name using valid characters and try again."""
        )
    return value


def _cluster_status_long(cluster: V1GetClusterResponse, desired_state: V1ClusterState, elapsed: float) -> str:
    """Echos a long-form status message to the user about the cluster state.

    Args:
        cluster: The cluster object
        elapsed: Seconds since we've started polling
    """

    cluster_id = cluster.id
    current_state = cluster.status.phase
    current_reason = cluster.status.reason
    bucket_name = cluster.spec.driver.kubernetes.aws.bucket_name

    duration = _format_elapsed_seconds(elapsed)

    if current_state == V1ClusterState.FAILED:
        if not _is_retryable_error(current_reason):
            return dedent(
                f"""\
                The requested cluster operation for cluster {cluster_id} has errors:

                {current_reason}

                --------------------------------------------------------------

                We are automatically retrying, and an automated alert has been created

                WARNING: Any non-deleted cluster may be using resources.
                To avoid incuring cost on your cloud provider, delete the cluster using the following command:
                    lightning delete cluster {cluster_id}

                Contact support@lightning.ai for additional help
                """
            )

    if desired_state == current_state == V1ClusterState.RUNNING:
        return dedent(
            f"""\
                Cluster {cluster_id} is now running and ready to use.
                To launch an app on this cluster use: lightning run app app.py --cloud --cluster-id {cluster_id}
                """
        )

    if desired_state == V1ClusterState.RUNNING:
        return f"Cluster {cluster_id} is being created [elapsed={duration}]"

    if desired_state == current_state == V1ClusterState.DELETED:
        return dedent(
            f"""\
            Cluster {cluster_id} has been successfully deleted, and almost all AWS resources have been removed

            For safety purposes we kept the S3 bucket associated with the cluster: {bucket_name}

            You may want to delete it manually using the AWS CLI:
                aws s3 rb --force s3://{bucket_name}
            """
        )

    if desired_state == V1ClusterState.DELETED:
        return f"Cluster {cluster_id} is being deleted [elapsed={duration}]"

    raise click.ClickException(f"Unknown cluster desired state {desired_state}")


def _is_retryable_error(error_message: str) -> bool:
    return "resources failed to delete" in error_message


def _format_elapsed_seconds(seconds: Union[float, int]) -> str:
    """Turns seconds into a duration string.

    >>> _format_elapsed_seconds(5)
    '05s'
    >>> _format_elapsed_seconds(60)
    '01m00s'
    """
    minutes, seconds = divmod(seconds, 60)
    return (f"{minutes:02}m" if minutes else "") + f"{seconds:02}s"
