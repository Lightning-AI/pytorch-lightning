import click

from lightning_app.cli.cmd_clusters import _check_cluster_name_is_valid, _default_instance_types, AWSClusterManager
from lightning_app.cli.lightning_cli import _main


@_main.group("create")
def create():
    """Create Lightning AI BYOC managed resources."""
    pass


@create.command("cluster")
@click.argument("cluster_name", callback=_check_cluster_name_is_valid)
@click.option("--provider", "provider", type=str, default="aws", help="cloud provider to be used for your cluster")
@click.option("--external-id", "external_id", type=str, required=True)
@click.option(
    "--role-arn", "role_arn", type=str, required=True, help="AWS role ARN attached to the associated resources."
)
@click.option(
    "--region",
    "region",
    type=str,
    required=False,
    default="us-east-1",
    help="AWS region that is used to host the associated resources.",
)
@click.option(
    "--instance-types",
    "instance_types",
    type=str,
    required=False,
    default=",".join(_default_instance_types),
    help="Instance types that you want to support, for computer jobs within the cluster.",
)
@click.option(
    "--cost-savings",
    "cost_savings",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help=""""Use this flag to ensure that the cluster is created with a profile that is optimized for cost savings.
        This makes runs cheaper but start-up times may increase.""",
)
@click.option(
    "--edit-before-creation",
    default=False,
    is_flag=True,
    help="Edit the cluster specs before submitting them to the API server.",
)
@click.option(
    "--wait",
    "wait",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Enabling this flag makes the CLI wait until the cluster is running.",
)
def create_cluster(
    cluster_name: str,
    region: str,
    role_arn: str,
    external_id: str,
    provider: str,
    instance_types: str,
    edit_before_creation: bool,
    cost_savings: bool,
    wait: bool,
    **kwargs,
):
    """Create a Lightning AI BYOC compute cluster with your cloud provider credentials."""
    if provider != "aws":
        click.echo("Only AWS is supported for now. But support for more providers is coming soon.")
        return
    cluster_manager = AWSClusterManager()
    cluster_manager.create(
        cluster_name=cluster_name,
        region=region,
        role_arn=role_arn,
        external_id=external_id,
        instance_types=instance_types.split(","),
        edit_before_creation=edit_before_creation,
        cost_savings=cost_savings,
        wait=wait,
    )
