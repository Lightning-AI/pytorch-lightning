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

from typing import Optional

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
    "--sync",
    "do_sync",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="This flag makes the CLI wait until cluster deletion completes.",
)
def delete_cluster(cluster: str, force: bool = False, do_sync: bool = False) -> None:
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
    cluster_manager.delete(cluster_id=cluster, force=force, do_async=not do_sync)


def _find_cluster_for_user(app_name: str, cluster_id: Optional[str]) -> str:
    console = Console()
    cluster_manager = AWSClusterManager()

    default_cluster, valid_cluster_list = None, []
    for cluster in cluster_manager.list_clusters():
        valid_cluster_list.append(cluster.id)
        if cluster.spec.cluster_type == V1ClusterType.GLOBAL and default_cluster is None:
            default_cluster = cluster.id

    # when no cluster-id is passed in, use the default (Lightning Cloud) cluster
    if cluster_id is None:
        cluster_id = default_cluster

    if cluster_id not in valid_cluster_list:
        console.print(f'[b][yellow]You don\'t have access to cluster "{cluster_id}"[/yellow][/b]')
        if len(valid_cluster_list) == 1:
            # if there are no BYOC clusters, then confirm that
            # the user wants to fall back to the Lightning Cloud.
            try:
                ask = [
                    inquirer.Confirm(
                        "confirm",
                        message=f'Did you mean to specify the default Lightning Cloud cluster "{default_cluster}"?',
                        default=True,
                    ),
                ]
                if inquirer.prompt(ask, theme=GreenPassion(), raise_keyboard_interrupt=True)["confirm"] is False:
                    console.print("[b][red]Aborted![/b][/red]")
                    raise InterruptedError
            except KeyboardInterrupt:
                console.print("[b][red]Cancelled by user![/b][/red]")
                raise InterruptedError
            cluster_id = default_cluster
        else:
            # When there are BYOC clusters, have them select which cluster to use from all available.
            try:
                ask = [
                    inquirer.List(
                        "cluster",
                        message=f'Please select which cluster app "{app_name}" should be deleted from',
                        choices=valid_cluster_list,
                        default=default_cluster,
                    ),
                ]
                cluster_id = inquirer.prompt(ask, theme=GreenPassion(), raise_keyboard_interrupt=True)["cluster"]
            except KeyboardInterrupt:
                console.print("[b][red]Cancelled by user![/b][/red]")
                raise InterruptedError

    if cluster_id is None:
        # stupid mypy thing...
        cluster_id = ""

    return cluster_id


def _find_selected_app_instance_id(app_name: str, cluster_id: str) -> str:
    console = Console()
    app_manager = _AppManager()

    all_app_names_and_ids = {}
    selected_app_instance_id = None

    for app in app_manager.list_apps(cluster_id=cluster_id):
        all_app_names_and_ids[app.name] = app.id
        # figure out the ID of some app_name on the cluster.
        if app_name == app.name or app_name == app.id:
            selected_app_instance_id = app.id
            break

    if selected_app_instance_id is None:
        # when there is no app with the given app_name on the cluster,
        # ask the user which app they would like to delete.
        console.print(f'[b][yellow]Cluster "{cluster_id}" does not have an app named "{app_name}"[/yellow][/b]')
        try:
            ask = [
                inquirer.List(
                    "app_name",
                    message="Select the app name to delete",
                    choices=list(all_app_names_and_ids.keys()),
                ),
            ]
            app_name = inquirer.prompt(ask, theme=GreenPassion(), raise_keyboard_interrupt=True)["app_name"]
            selected_app_instance_id = all_app_names_and_ids[app_name]
        except KeyboardInterrupt:
            console.print("[b][red]Cancelled by user![/b][/red]")
            raise InterruptedError

    return selected_app_instance_id


def _delete_app_confirmation_prompt(app_name: str, cluster_id: str) -> None:
    console = Console()

    # when the --yes / -y flags were not passed, do a final
    # confirmation that the user wants to delete the app.
    try:
        ask = [
            inquirer.Confirm(
                "confirm",
                message=f'Are you sure you want to delete app "{app_name}" on cluster "{cluster_id}"?',
                default=False,
            ),
        ]
        if inquirer.prompt(ask, theme=GreenPassion(), raise_keyboard_interrupt=True)["confirm"] is False:
            console.print("[b][red]Aborted![/b][/red]")
            raise InterruptedError
    except KeyboardInterrupt:
        console.print("[b][red]Cancelled by user![/b][/red]")
        raise InterruptedError


@delete.command("app")
@click.argument("app-name", type=str)
@click.option(
    "--cluster-id",
    type=str,
    default=None,
    help="Delete the Lighting App from a specific Lightning AI BYOC compute cluster",
)
@click.option(
    "skip_user_confirm_prompt",
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Do not prompt for confirmation.",
)
def delete_app(app_name: str, cluster_id: str, skip_user_confirm_prompt: bool) -> None:
    """Delete a Lightning app.

    Deleting an app also deletes all app websites, works, artifacts, and logs. This permanently removes any record of
    the app as well as all any of its associated resources and data. This does not affect any resources and data
    associated with other Lightning apps on your account.
    """
    console = Console()

    try:
        cluster_id = _find_cluster_for_user(app_name=app_name, cluster_id=cluster_id)
        selected_app_instance_id = _find_selected_app_instance_id(app_name=app_name, cluster_id=cluster_id)
        if not skip_user_confirm_prompt:
            _delete_app_confirmation_prompt(app_name=app_name, cluster_id=cluster_id)
    except InterruptedError:
        return

    try:
        # Delete the app!
        app_manager = _AppManager()
        app_manager.delete(app_id=selected_app_instance_id)
    except Exception as e:
        console.print(
            f'[b][red]An issue occurred while deleting app "{app_name}. If the issue persists, please '
            "reach out to us at [link=mailto:support@lightning.ai]support@lightning.ai[/link][/b][/red]."
        )
        raise click.ClickException(str(e))

    console.print(f'[b][green]App "{app_name}" has been successfully deleted from cluster "{cluster_id}"![/green][/b]')
    return


@delete.command("ssh-key")
@click.argument("key_id")
def remove_ssh_key(key_id: str) -> None:
    """Delete a ssh-key from your Lightning AI account."""
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.remove_key(key_id=key_id)
