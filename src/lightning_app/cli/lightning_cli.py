import os
import sys
from pathlib import Path
from typing import Any, List, Tuple, Union

import arrow
import click
import rich
from lightning_cloud.openapi import Externalv1LightningappInstance
from requests.exceptions import ConnectionError
from rich.color import ANSI_COLOR_NAMES

from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.cli.cmd_clusters import AWSClusterManager
from lightning_app.cli.commands.app_commands import _run_app_command
from lightning_app.cli.commands.connection import (
    _list_app_commands,
    _retrieve_connection_to_an_app,
    connect,
    disconnect,
)
from lightning_app.cli.lightning_cli_create import create
from lightning_app.cli.lightning_cli_delete import delete
from lightning_app.cli.lightning_cli_list import get_list
from lightning_app.core.constants import DEBUG, get_lightning_cloud_url
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.app_logs import _app_logs_reader
from lightning_app.utilities.cli_helpers import _arrow_time_callback, _format_input_env_variables
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.cluster_logs import _cluster_logs_reader
from lightning_app.utilities.exceptions import LogLinesLimitExceeded
from lightning_app.utilities.login import Auth
from lightning_app.utilities.logs_socket_api import _ClusterLogsSocketAPI, _LightningLogsSocketAPI
from lightning_app.utilities.network import LightningClient

logger = Logger(__name__)


def get_app_url(runtime_type: RuntimeType, *args: Any, need_credits: bool = False) -> str:
    if runtime_type == RuntimeType.CLOUD:
        lightning_app: Externalv1LightningappInstance = args[0]
        action = "?action=add_credits" if need_credits else ""
        return f"{get_lightning_cloud_url()}/me/apps/{lightning_app.id}{action}"
    else:
        return "http://127.0.0.1:7501/view"


def main() -> None:
    # 1: Handle connection to a Lightning App.
    if len(sys.argv) > 1 and sys.argv[1] in ("connect", "disconnect"):
        _main()
    else:
        # 2: Collect the connection a Lightning App.
        app_name, app_id = _retrieve_connection_to_an_app()
        if app_name:
            # 3: Handle development use case.
            is_local_app = app_name == "localhost"
            if sys.argv[1:3] == ["run", "app"] or sys.argv[1:3] == ["show", "logs"]:
                _main()
            else:
                if is_local_app:
                    click.echo("You are connected to the local Lightning App.")
                else:
                    click.echo(f"You are connected to the cloud Lightning App: {app_name}.")

                if "help" in sys.argv[1]:
                    _list_app_commands()
                else:
                    _run_app_command(app_name, app_id)
        else:
            _main()


@click.group()
@click.version_option(ver)
def _main() -> None:
    pass


@_main.group()
def show() -> None:
    """Show given resource."""
    pass


_main.command()(connect)
_main.command()(disconnect)


@show.command()
@click.argument("app_name", required=False)
@click.argument("components", nargs=-1, required=False)
@click.option("-f", "--follow", required=False, is_flag=True, help="Wait for new logs, to exit use CTRL+C.")
def logs(app_name: str, components: List[str], follow: bool) -> None:
    """Show cloud application logs. By default prints logs for all currently available components.

    Example uses:

        Print all application logs:

            $ lightning show logs my-application


        Print logs only from the flow (no work):

            $ lightning show logs my-application flow


        Print logs only from selected works:

            $ lightning show logs my-application root.work_a root.work_b
    """

    client = LightningClient()
    project = _get_project(client)

    apps = {
        app.name: app
        for app in client.lightningapp_instance_service_list_lightningapp_instances(
            project_id=project.project_id
        ).lightningapps
    }

    if not apps:
        raise click.ClickException(
            "You don't have any application in the cloud. Please, run an application first with `--cloud`."
        )

    if not app_name:
        raise click.ClickException(
            f"You have not specified any Lightning App. Please select one of available: [{', '.join(apps.keys())}]"
        )

    if app_name not in apps:
        raise click.ClickException(
            f"The Lightning App '{app_name}' does not exist. Please select one of following: [{', '.join(apps.keys())}]"
        )

    # Fetch all lightning works from given application
    # 'Flow' component is somewhat implicit, only one for whole app,
    #    and not listed in lightningwork API - so we add it directly to the list
    works = client.lightningwork_service_list_lightningwork(
        project_id=project.project_id, app_id=apps[app_name].id
    ).lightningworks
    app_component_names = ["flow"] + [f.name for f in apps[app_name].spec.flow_servers] + [w.name for w in works]

    if not components:
        components = app_component_names

    else:

        def add_prefix(c: str) -> str:
            if c == "flow":
                return c
            if not c.startswith("root."):
                return "root." + c
            return c

        components = [add_prefix(c) for c in components]

        for component in components:
            if component not in app_component_names:
                raise click.ClickException(f"Component '{component}' does not exist in app {app_name}.")

    log_reader = _app_logs_reader(
        logs_api_client=_LightningLogsSocketAPI(client.api_client),
        project_id=project.project_id,
        app_id=apps[app_name].id,
        component_names=components,
        follow=follow,
    )

    rich_colors = list(ANSI_COLOR_NAMES)
    colors = {c: rich_colors[i + 1] for i, c in enumerate(components)}

    for log_event in log_reader:
        date = log_event.timestamp.strftime("%m/%d/%Y %H:%M:%S")
        color = colors[log_event.component_name]
        rich.print(f"[{color}]{log_event.component_name}[/{color}] {date} {log_event.message}")


@show.group()
def cluster() -> None:
    """Groups cluster commands inside show."""
    pass


@cluster.command(name="logs")
@click.argument("cluster_name", required=True)
@click.option(
    "--from",
    "from_time",
    default="24 hours ago",
    help="The starting timestamp to query cluster logs from. Human-readable (e.g. '48 hours ago') or ISO 8601 "
    "(e.g. '2022-08-23 12:34') formats.",
    callback=_arrow_time_callback,
)
@click.option(
    "--to",
    "to_time",
    default="0 seconds ago",
    callback=_arrow_time_callback,
    help="The end timestamp / relative time increment to query logs for. This is ignored when following logs (with "
    "-f/--follow). The same format as --from option has.",
)
@click.option("--limit", default=10000, help="The max number of log lines returned.")
@click.option("-f", "--follow", required=False, is_flag=True, help="Wait for new logs, to exit use CTRL+C.")
def cluster_logs(cluster_name: str, to_time: arrow.Arrow, from_time: arrow.Arrow, limit: int, follow: bool) -> None:
    """Show cluster logs.

    Example uses:

        Print cluster logs:

            $ lightning show cluster logs my-cluster


        Print cluster logs and wait for new logs:

            $ lightning show cluster logs my-cluster --follow


        Print cluster logs, from 48 hours ago to now:

            $ lightning show cluster logs my-cluster --from "48 hours ago"


        Print cluster logs, 10 most recent lines:

            $ lightning show cluster logs my-cluster --limit 10
    """

    client = LightningClient()
    cluster_manager = AWSClusterManager()
    existing_cluster_list = cluster_manager.get_clusters()

    clusters = {cluster.name: cluster.id for cluster in existing_cluster_list.clusters}

    if not clusters:
        raise click.ClickException("You don't have any clusters.")

    if not cluster_name:
        raise click.ClickException(
            f"You have not specified any clusters. Please select one of available: [{', '.join(clusters.keys())}]"
        )

    if cluster_name not in clusters:
        raise click.ClickException(
            f"The cluster '{cluster_name}' does not exist."
            f" Please select one of the following: [{', '.join(clusters.keys())}]"
        )

    try:
        log_reader = _cluster_logs_reader(
            logs_api_client=_ClusterLogsSocketAPI(client.api_client),
            cluster_id=clusters[cluster_name],
            start=from_time.int_timestamp,
            end=to_time.int_timestamp if not follow else None,
            limit=limit,
            follow=follow,
        )

        colors = {"error": "red", "warn": "yellow", "info": "green"}

        for log_event in log_reader:
            date = log_event.timestamp.strftime("%m/%d/%Y %H:%M:%S")
            color = colors.get(log_event.labels.level, "green")
            rich.print(f"[{color}]{log_event.labels.level:5}[/{color}] {date} {log_event.message.rstrip()}")
    except LogLinesLimitExceeded:
        raise click.ClickException(f"Read {limit} log lines, but there may be more. Use --limit param to read more")
    except Exception as error:
        logger.error(f"⚡ Error while reading logs ({type(error)}), {error}", exc_info=DEBUG)


@_main.command()
def login() -> None:
    """Log in to your lightning.ai account."""
    auth = Auth()
    auth.clear()

    try:
        auth._run_server()
    except ConnectionError:
        click.echo(f"Unable to connect to {get_lightning_cloud_url()}. Please check your internet connection.")
        exit(1)


@_main.command()
def logout() -> None:
    """Log out of your lightning.ai account."""
    Auth().clear()
    disconnect(logout=True)


def _run_app(
    file: str,
    cloud: bool,
    cluster_id: str,
    without_server: bool,
    no_cache: bool,
    name: str,
    blocking: bool,
    open_ui: bool,
    env: tuple,
    secret: tuple,
) -> None:
    file = _prepare_file(file)

    if not cloud and cluster_id is not None:
        raise click.ClickException("Using the flag --cluster-id in local execution is not supported.")

    runtime_type = RuntimeType.CLOUD if cloud else RuntimeType.MULTIPROCESS

    # Cloud specific validations
    if runtime_type != RuntimeType.CLOUD:
        if no_cache:
            raise click.ClickException(
                "Caching is a property of apps running in cloud. "
                "Using the flag --no-cache in local execution is not supported."
            )
        if secret:
            raise click.ClickException(
                "Secrets can only be used for apps running in cloud. "
                "Using the option --secret in local execution is not supported."
            )

    env_vars = _format_input_env_variables(env)
    os.environ.update(env_vars)

    secrets = _format_input_env_variables(secret)

    def on_before_run(*args: Any, **kwargs: Any) -> None:
        if open_ui and not without_server:
            click.launch(get_app_url(runtime_type, *args, **kwargs))

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
        secrets=secrets,
        cluster_id=cluster_id,
    )
    if runtime_type == RuntimeType.CLOUD:
        click.echo("Application is ready in the cloud")


@_main.group()
def run() -> None:
    """Run a Lightning application locally or on the cloud."""


@run.command("app")
@click.argument("file", type=click.Path(exists=True))
@click.option("--cloud", type=bool, default=False, is_flag=True)
@click.option(
    "--cluster-id",
    type=str,
    default=None,
    help="Run Lightning App on a specific Lightning AI BYOC compute cluster",
)
@click.option("--name", help="The current application name", default="", type=str)
@click.option("--without-server", is_flag=True, default=False)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching of packages " "installed from requirements.txt",
)
@click.option("--blocking", "blocking", type=bool, default=False)
@click.option(
    "--open-ui",
    type=bool,
    default=True,
    help="Decide whether to launch the app UI in a web browser",
)
@click.option("--env", type=str, default=[], multiple=True, help="Environment variables to be set for the app.")
@click.option("--secret", type=str, default=[], multiple=True, help="Secret variables to be set for the app.")
@click.option("--app_args", type=str, default=[], multiple=True, help="Collection of arguments for the app.")
def run_app(
    file: str,
    cloud: bool,
    cluster_id: str,
    without_server: bool,
    no_cache: bool,
    name: str,
    blocking: bool,
    open_ui: bool,
    env: tuple,
    secret: tuple,
    app_args: tuple,
) -> None:
    """Run an app from a file."""
    _run_app(file, cloud, cluster_id, without_server, no_cache, name, blocking, open_ui, env, secret)


@_main.group(hidden=True)
def fork() -> None:
    """Fork an application."""
    pass


@_main.group(hidden=True)
def stop() -> None:
    """Stop your application."""
    pass


_main.add_command(get_list)
_main.add_command(delete)
_main.add_command(create)


@_main.group()
def install() -> None:
    """Install a Lightning App and/or component."""


@install.command("app")
@click.argument("name", type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="disables prompt to ask permission to create env and run install cmds",
)
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
def install_app(name: str, yes: bool, version: str, overwrite: bool = False) -> None:
    if "github.com" in name:
        if version != "latest":
            logger.warn(
                f"The provided version {version} isn't the officially supported one. "
                f"The provided version will be ignored."
            )
        cmd_install.non_gallery_app(name, yes, overwrite=overwrite)
    else:
        cmd_install.gallery_app(name, yes, version, overwrite=overwrite)


@install.command("component")
@click.argument("name", type=str)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="disables prompt to ask permission to create env and run install cmds",
)
@click.option(
    "--version",
    "-v",
    type=str,
    help="Specify the version to install. By default it uses 'latest'",
    default="latest",
    show_default=True,
)
def install_component(name: str, yes: bool, version: str) -> None:
    if "github.com" in name:
        if version != "latest":
            logger.warn(
                f"The provided version {version} isn't the officially supported one. "
                f"The provided version will be ignored."
            )
        cmd_install.non_gallery_component(name, yes)
    else:
        cmd_install.gallery_component(name, yes, version)


@_main.group()
def init() -> None:
    """Init a Lightning App and/or component."""


@init.command("app")
@click.argument("name", type=str, required=False)
def init_app(name: str) -> None:
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
        # enable type checking once https://github.com/python/mypy/issues/1178 is available
        source_dir, script_path = source  # type: ignore
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
def init_component(name: str) -> None:
    cmd_init.component(name)


@init.command("react-ui")
@click.option(
    "--dest_dir",
    "-dest_dir",
    type=str,
    help="optional destination directory to create the react ui",
)
def init_react_ui(dest_dir: str) -> None:
    """Create a react UI to give a Lightning component a React.js web user interface (UI)"""
    cmd_react_ui_init.react_ui(dest_dir)


def _prepare_file(file: str) -> str:
    exists = os.path.exists(file)
    if exists:
        return file

    raise FileNotFoundError(f"The provided file {file} hasn't been found.")
