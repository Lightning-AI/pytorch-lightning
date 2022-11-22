import click

from lightning_app.cli.cmd_clusters import AWSClusterManager
from lightning_app.cli.cmd_ssh_keys import _SSHKeyManager


@click.group("delete")
def delete() -> None:
    """Delete Lightning AI self-managed resources (clusters, etc…)"""
    pass


@delete.command("cluster")
@click.argument("cluster", type=str)
@click.option(
    "--force",
    "force",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="""Delete a BYOC cluster from Lightning AI. This does NOT delete any resources created by the cluster,
            it just removes the entry from Lightning AI.

            WARNING: You should NOT use this under normal circumstances.""",
)
@click.option(
    "--wait",
    "wait",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Enabling this flag makes the CLI wait until the cluster is deleted.",
)
def delete_cluster(cluster: str, force: bool = False, wait: bool = False) -> None:
    """Delete a Lightning AI BYOC cluster and all associated cloud provider resources.

    Deleting a cluster also deletes all apps that were started on the cluster.
    Deletion permanently removes not only the record of all apps run on a cluster,
    but all associated data, artifacts, metrics, logs, web-UIs, etc.

    WARNING: This process may take a few minutes to complete, but once started it
    CANNOT be rolled back. Deletion tears down every cloud provider resource
    managed by Lightning AI and permanently revokes the ability for Lightning AI
    to create, manage, or access any resources within the host cloud account.

    All object stores, container registries, logs, compute nodes, volumes,
    VPC components, etc. are irreversibly deleted and cannot be recovered!
    """
    cluster_manager = AWSClusterManager()
    cluster_manager.delete(cluster_id=cluster, force=force, wait=wait)


@delete.command("ssh-key")
@click.argument("key_id")
def remove_ssh_key(key_id: str) -> None:
    """Delete a ssh-key from your Lightning AI account."""
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.remove_key(key_id=key_id)
