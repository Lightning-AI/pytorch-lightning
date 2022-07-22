import logging
import os
import re
import json
import time
import click
from pathlib import Path
from typing import List, Tuple, Union

from requests.exceptions import ConnectionError

from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.core.constants import get_lightning_cloud_url, LOCAL_LAUNCH_ADMIN_VIEW
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.utilities.openapi import create_openapi_object, string2dict
from lightning_app.utilities.cli_helpers import _format_input_env_variables
from lightning_app.utilities.install_components import register_all_external_components
from lightning_app.utilities.login import Auth
from lightning_app.utilities.network import LightningClient
from lightning_cloud.openapi.models import (
    V1ClusterState,
    V1CreateClusterRequest,
    V1AWSClusterDriverSpec,
    V1InstanceSpec,
    V1ClusterSpec,
    V1ClusterType,
    V1ClusterPerformanceProfile,
    V1ClusterDriver,
    V1KubernetesClusterDriver,
)
from lightning_app.cli.cmd_clusters import ClusterList
from rich.console import Console


logger = logging.getLogger(__name__)


def get_app_url(runtime_type: RuntimeType, *args) -> str:
    if runtime_type == RuntimeType.CLOUD:
        lightning_app = args[0]
        return f"{get_lightning_cloud_url()}/me/apps/{lightning_app.id}"
    else:
        return "http://127.0.0.1:7501/admin" if LOCAL_LAUNCH_ADMIN_VIEW else "http://127.0.0.1:7501/view"


@click.group()
def clusters():
    """Manage your Lightning.ai BYOC clusters"""
    pass


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
CLUSTER_STATE_CHECKING_TIMEOUT = 60
MAX_CLUSTER_WAIT_TIME = 5400


@clusters.command('create')
@click.argument('cluster_name', callback=_check_cluster_name_is_valid)
@click.option("--provider", 'provider', type=str, default="aws", help="cloud provider to be used for your cluster")
@click.option('--external-id', 'external_id', type=str, required=True)
@click.option(
    '--role-arn', 'role_arn', type=str, required=True, help="AWS role ARN attached to`the associated resources."
)
@click.option(
    '--region',
    'region',
    type=str,
    required=False,
    default="us-east-1",
    help="AWS region which is used to host the associated resources."
)
@click.option(
    '--instance-types',
    'instance_types',
    type=str,
    required=False,
    default=",".join(default_instance_types),
    help="Instance types which you desire to support for computer jobs within the cluster."
)
@click.option(
    '--cost-savings',
    'cost_savings',
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help='using this flag ensures that the cluster is created with a profile that is optimized for '
         'cost saving, making runs cheaper but start-up times may increase',
)
@click.option(
    '--edit-before-creation',
    default=False,
    is_flag=True,
    help="Edit the created cluster spec before submitting to API server."
)
@click.option(
    '--wait',
    'wait',
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help='using this flag CLI will wait until the cluster is running',
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
        **kwargs):
    """Create a Lightning.ai BYOC clusters"""
    if provider != "aws":
        click.echo("only AWS is supported today")
        return

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
    api_client = LightningClient()
    resp = api_client.cluster_service_create_cluster(body=new_body)
    if wait:
        start = time.time()
        elapsed = 0
        while elapsed < MAX_CLUSTER_WAIT_TIME:
            cluster_resp = api_client.cluster_service_list_clusters(phase_not_in=[V1ClusterState.DELETED])
            new_cluster = None
            for clust in cluster_resp.clusters:
                if clust.id == resp.id:
                    new_cluster = clust
                    break
            if new_cluster is not None:
                if new_cluster.status.phase == V1ClusterState.RUNNING:
                    break
                elif new_cluster.status.phase == V1ClusterState.FAILED:
                    raise click.ClickException(f"Creation failed for cluster {resp.id}")
                time.sleep(CLUSTER_STATE_CHECKING_TIMEOUT)
            elapsed = time.time() - start
        else:
            raise click.ClickException(f"Max time for cluster creation is elapsed")

    click.echo(resp.to_str())


@clusters.command('list')
def list_clusters(**kwargs):
    api_client = LightningClient()
    resp = api_client.cluster_service_list_clusters(phase_not_in=[V1ClusterState.DELETED])
    console = Console()
    console.print(ClusterList(resp.clusters).as_table())
    """List your Lightning.ai BYOC clusters"""
    click.echo('TODO(rra) list clusters')
    pass


@clusters.command('delete')
@click.argument('cluster', type=str)
@click.option(
    '--force',
    'force',
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help='Force delete cluster from grid system. This does NOT delete any resources created by the cluster, '
         'just cleaning up the entry from the grid system. You should not use this under normal circumstances',
)
@click.option(
    '--wait',
    'wait',
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help='using this flag CLI will wait until the cluster is deleted',
)
def delete_cluster(cluster: str, force: bool = False, wait: bool = False):
    """Delete CLUSTER and all associated AWS resources.

    Deleting a run also deletes all Runs and Experiments which were started
    on the cluster. deletion permanently removes not only the record of all
    runs on a cluster, but all associated experiments, artifacts, metrics, logs, etc.

    This process may take a few minutes to complete, but once started is irriversable.
    Deletion permanently removes not only cluster from being managed by Lightning.ai, but tears
    down every resource Lightning managed (for that cluster id) in the host cloud. All object
    stores, container registries, logs, compute nodes, volumes, etc. are deleted and
    cannot be recovered.
    """
    if force:
        click.echo(
            "Force deleting cluster. This will cause grid to forget "
            "about the cluster and any experiments, sessions, datastores, "
            "tensorboards and other resources running on it.\n"
            "WARNING: this will not clean up any resources managed by grid\n"
            "Check your cloud provider that any existing cloud resources are deleted"
        )
        click.confirm('Do you want to continue?', abort=True)

    client = LightningClient()
    client.cluster_service_delete_cluster(id=cluster, force=force)
    click.echo("Cluster deletion triggered successfully")

    if wait:
        start = time.time()
        elapsed = 0
        while elapsed < MAX_CLUSTER_WAIT_TIME:
            cluster_resp = client.cluster_service_list_clusters()
            cluster_to_del = None
            for clust in cluster_resp.clusters:
                if clust.id == cluster:
                    cluster_to_del = clust
                    break
            if cluster_to_del is not None:
                if cluster_to_del.status.phase == V1ClusterState.DELETED:
                    break
                elif cluster_to_del.status.phase == V1ClusterState.FAILED:
                    raise click.ClickException(f"Deletion failed for cluster {cluster}")
                time.sleep(CLUSTER_STATE_CHECKING_TIMEOUT)
            else:
                break
            elapsed = time.time() - start
        else:
            raise click.ClickException(f"Max time for cluster deletion is elapsed")



@click.group()
@click.version_option(ver)
def main():
    register_all_external_components()
    pass


main.add_command(clusters)


@main.command()
def login():
    """Log in to your Lightning.ai account."""
    auth = Auth()
    auth.clear()

    try:
        auth._run_server()
    except ConnectionError:
        click.echo(f"Unable to connect to {get_lightning_cloud_url()}. Please check your internet connection.")
        exit(1)


@main.command()
def logout():
    """Log out of your Lightning.ai account."""
    Auth().clear()


def _run_app(
    file: str, cloud: bool, without_server: bool, no_cache: bool, name: str, blocking: bool, open_ui: bool, env: tuple
):
    file = _prepare_file(file)

    runtime_type = RuntimeType.CLOUD if cloud else RuntimeType.MULTIPROCESS

    # Cloud specific validations
    if runtime_type != RuntimeType.CLOUD:
        if no_cache:
            raise click.ClickException(
                "Caching is a property of apps running in cloud. "
                "Using the flag --no-cache in local execution is not supported."
            )

    env_vars = _format_input_env_variables(env)
    os.environ.update(env_vars)

    def on_before_run(*args):
        if open_ui and not without_server:
            click.launch(get_app_url(runtime_type, *args))

    click.echo("Your Lightning App is starting. This won't take long.")

    # TODO: Fixme when Grid utilities are available.
    # And refactor test_lightning_run_app_cloud
    file_path = Path(file)
    dispatch(
        file_path,
        runtime_type,
        start_server=not without_server,
        no_cache=no_cache,
        blocking=blocking,
        on_before_run=on_before_run,
        name=name,
        env_vars=env_vars,
    )
    if runtime_type == RuntimeType.CLOUD:
        click.echo("Application is ready in the cloud")


@main.group()
def run():
    """Run your application."""


@run.command("app")
@click.argument("file", type=click.Path(exists=True))
@click.option("--cloud", type=bool, default=False, is_flag=True)
@click.option("--name", help="The current application name", default="", type=str)
@click.option("--without-server", is_flag=True, default=False)
@click.option(
    "--no-cache", is_flag=True, default=False, help="Disable caching of packages " "installed from requirements.txt"
)
@click.option("--blocking", "blocking", type=bool, default=False)
@click.option("--open-ui", type=bool, default=True, help="Decide whether to launch the app UI in a web browser")
@click.option("--env", type=str, default=[], multiple=True, help="Env variables to be set for the app.")
@click.option("--app_args", type=str, default=[], multiple=True, help="Collection of arguments for the app.")
def run_app(
    file: str,
    cloud: bool,
    without_server: bool,
    no_cache: bool,
    name: str,
    blocking: bool,
    open_ui: bool,
    env: tuple,
    app_args: List[str],
):
    """Run an app from a file."""
    _run_app(file, cloud, without_server, no_cache, name, blocking, open_ui, env)


@main.group(hidden=True)
def fork():
    """Fork an application."""
    pass


@main.group(hidden=True)
def stop():
    """Stop your application."""
    pass


@main.group(hidden=True)
def delete():
    """Delete an application."""
    pass


@main.group(name="list", hidden=True)
def get_list():
    """List your applications."""
    pass


@main.group()
def install():
    """Install Lightning apps and components."""


@install.command("app")
@click.argument("name", type=str)
@click.option("--yes", "-y", is_flag=True, help="disables prompt to ask permission to create env and run install cmds")
@click.option(
    "--version",
    "-v",
    type=str,
    help="Specify the version to install. By default it uses 'latest'",
    default="latest",
    show_default=True,
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    help="When set, overwrite the app directory without asking if it already exists.",
)
def install_app(name, yes, version, overwrite: bool = False):
    if "github.com" in name:
        if version != "latest":
            logger.warning(
                f"The provided version {version} isn't the officially supported one. "
                f"The provided version will be ignored."
            )
        cmd_install.non_gallery_app(name, yes, overwrite=overwrite)
    else:
        cmd_install.gallery_app(name, yes, version, overwrite=overwrite)


@install.command("component")
@click.argument("name", type=str)
@click.option("--yes", "-y", is_flag=True, help="disables prompt to ask permission to create env and run install cmds")
@click.option(
    "--version",
    "-v",
    type=str,
    help="Specify the version to install. By default it uses 'latest'",
    default="latest",
    show_default=True,
)
def install_component(name, yes, version):
    if "github.com" in name:
        if version != "latest":
            logger.warning(
                f"The provided version {version} isn't the officially supported one. "
                f"The provided version will be ignored."
            )
        cmd_install.non_gallery_component(name, yes)
    else:
        cmd_install.gallery_component(name, yes, version)


@main.group()
def init():
    """Init a Lightning app and component."""


@init.command("app")
@click.argument("name", type=str, required=False)
def init_app(name):
    cmd_init.app(name)


@init.command("pl-app")
@click.argument("source", nargs=-1)
@click.option(
    "--name",
    "-n",
    type=str,
    default="pl-app",
    help="The name of the folder where the app code will be. Default: pl-app",
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    help="When set, overwrite the output directory without asking if it already exists.",
)
def init_pl_app(source: Union[Tuple[str], Tuple[str, str]], name: str, overwrite: bool = False) -> None:
    """Create an app from your PyTorch Lightning source files."""
    if len(source) == 1:
        script_path = source[0]
        source_dir = str(Path(script_path).resolve().parent)
    elif len(source) == 2:
        source_dir, script_path = source
    else:
        click.echo(
            f"Incorrect number of arguments. You passed ({', '.join(source)}) but only either one argument"
            f" (script path) or two arguments (root dir, script path) are allowed. Examples:\n"
            f"lightning init pl-app ./path/to/script.py\n"
            f"lightning init pl-app ./code ./code/path/to/script.py",
            err=True,
        )
        raise SystemExit(1)

    cmd_pl_init.pl_app(source_dir=source_dir, script_path=script_path, name=name, overwrite=overwrite)


@init.command("component")
@click.argument("name", type=str, required=False)
def init_component(name):
    cmd_init.component(name)


@init.command("react-ui")
@click.option("--dest_dir", "-dest_dir", type=str, help="optional destination directory to create the react ui")
def init_react_ui(dest_dir):
    """Create a react UI to give a Lightning component a React.js web user interface (UI)"""
    cmd_react_ui_init.react_ui(dest_dir)


def _prepare_file(file: str) -> str:
    exists = os.path.exists(file)
    if exists:
        return file

    raise FileNotFoundError(f"The provided file {file} hasn't been found.")
