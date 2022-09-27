import click

from lightning_app.cli.cmd_clusters import AWSClusterManager


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
    """Delete a Lightning AI BYOC compute cluster and all associated cloud provider resources.

    Deleting a run also deletes all Runs and Experiments that were started on the cluster.
    Deletion permanently removes not only the record of all runs on a cluster, but all associated experiments,
    artifacts, metrics, logs, etc.

    WARNING: This process may take a few minutes to complete, but once started it CANNOT be rolled back.
    Deletion permanently removes not only the BYOC cluster from being managed by Lightning AI, but tears down
    every BYOC resource Lightning AI managed (for that cluster id) in the host cloud.

    All object stores, container registries, logs, compute nodes, volumes, etc. are deleted and cannot be recovered.
    """
    cluster_manager = AWSClusterManager()
    cluster_manager.delete(cluster_id=cluster, force=force, wait=wait)
