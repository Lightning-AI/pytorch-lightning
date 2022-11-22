import click
import inquirer
from inquirer.themes import GreenPassion
from lightning_cloud.openapi import V1ClusterType
from rich.console import Console

from lightning_app.cli.cmd_apps import _AppManager
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


@delete.command("app")
@click.argument("app-name", type=str)
@click.option(
    "--cluster-id",
    type=str,
    default=None,
    help="Delete the Lighting App from a specific Lightning AI BYOC compute cluster",
)
def delete_app(app_name: str, cluster_id: str) -> None:
    """Delete a Lightning AI app and associated data and resources.

    Deleting an app also deletes all app websites, works, artifacts, and logs. This permanently removes not only the
    record of the app, but all resources associated with the app.
    """
    console = Console()
    cluster_manager = AWSClusterManager()
    app_manager = _AppManager()

    default_cluster, valid_clusters = None, []
    for cluster in cluster_manager.get_clusters().clusters:
        valid_clusters.append(cluster.id)
        if cluster.spec.cluster_type == V1ClusterType.GLOBAL and default_cluster is None:
            default_cluster = cluster.id

    if cluster_id is None:
        cluster_id = default_cluster

    else:
        if cluster_id not in valid_clusters:
            console.print(f"[warning][b][yellow]You don't have access to cluster: {cluster_id}[/yellow][/b][/warning]")
            if len(valid_clusters) == 1:
                confirm_cluster = [
                    inquirer.Confirm(
                        "confirm_cluster",
                        message=f"Did you mean to specify the default Lightning Cloud cluster: {default_cluster}?",
                        default=True,
                    ),
                ]
                if not inquirer.prompt(confirm_cluster, theme=GreenPassion())["confirm_cluster"]:
                    console.print("[b][yellow]Exiting![/b][/yellow]")
                    return
                cluster_id = default_cluster
            else:
                available_clusters = [
                    inquirer.List(
                        "cluster_id",
                        message="Please select which cluster the app should be deleted from",
                        choices=valid_clusters,
                        default=default_cluster,
                    ),
                ]
                cluster_id = inquirer.prompt(available_clusters, theme=GreenPassion())["cluster_id"]

    selected_app_instance_id = None
    all_app_names_and_ids = {}
    for app in app_manager.list_apps(cluster_id=cluster_id):
        all_app_names_and_ids[app.name] = app.id
        if app_name == app.name:
            selected_app_instance_id = app.id
            break
        if app_name == app.id:
            selected_app_instance_id = app.id
            break

    if selected_app_instance_id is None:
        console.print(
            f"[warning][b][yellow]Cluster {cluster_id} does not have an app named {app_name}[/yellow][/b][/warning]"
        )
        available_apps = [
            inquirer.List(
                "app_name",
                message="Select the app name to delete",
                choices=list(all_app_names_and_ids.keys()),
            ),
        ]
        app_name = inquirer.prompt(available_apps, theme=GreenPassion())["app_name"]
        selected_app_instance_id = all_app_names_and_ids[app_name]

    console.print(
        f"App {app_name} (debug id {selected_app_instance_id}) has been successfully deleted from cluster {cluster_id}"
    )
    return


@delete.command("ssh-key")
@click.argument("key_id")
def remove_ssh_key(key_id: str) -> None:
    """Delete a ssh-key from your Lightning AI account."""
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.remove_key(key_id=key_id)
