import click

from lightning_app.cli.cmd_clusters import AWSClusterManager


@click.group("delete")
def delete() -> None:
    """Delete Lightning AI self-managed resources (clusters, etc…)"""
    pass


@delete.command("cluster")
@click.argument("cluster", type=str)
@click.option(
    "--async",
    "do_async",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="This flag makes the CLI return immediately and lets the cluster deletion happen in the background",
)
def delete_cluster(cluster: str, do_async: bool = False) -> None:
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
    cluster_manager.delete(cluster_id=cluster, force=False, do_async=do_async)
