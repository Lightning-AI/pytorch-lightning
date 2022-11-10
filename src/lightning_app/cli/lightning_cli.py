import os
import shutil
import sys
from pathlib import Path
from typing import Any, Tuple, Union

import arrow
import click
import inquirer
import rich
from lightning_cloud.openapi import Externalv1LightningappInstance, V1LightningappInstanceState
from lightning_cloud.openapi.rest import ApiException
from requests.exceptions import ConnectionError

from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.cli.cmd_apps import _AppManager
from lightning_app.cli.cmd_clusters import AWSClusterManager
from lightning_app.cli.commands.app_commands import _run_app_command
from lightning_app.cli.commands.connection import (
    _list_app_commands,
    _retrieve_connection_to_an_app,
    connect,
    disconnect,
)
from lightning_app.cli.commands.logs import logs
from lightning_app.cli.lightning_cli_add import cli_add
from lightning_app.cli.lightning_cli_create import create
from lightning_app.cli.lightning_cli_delete import delete
from lightning_app.cli.lightning_cli_list import get_list
from lightning_app.cli.lightning_cli_remove import cli_remove
from lightning_app.core.constants import DEBUG, ENABLE_APP_COMMENT_COMMAND_EXECUTION, get_lightning_cloud_url
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.utilities.app_commands import run_app_commands
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cli_helpers import (
    _arrow_time_callback,
    _check_environment_and_redirect,
    _check_version_and_upgrade,
    _format_input_env_variables,
)
from lightning_app.utilities.cluster_logs import _cluster_logs_reader
from lightning_app.utilities.exceptions import _ApiExceptionHandler, LogLinesLimitExceeded
from lightning_app.utilities.login import Auth
from lightning_app.utilities.logs_socket_api import _ClusterLogsSocketAPI
from lightning_app.utilities.network import LightningClient

logger = Logger(__name__)


def get_app_url(runtime_type: RuntimeType, *args: Any, need_credits: bool = False) -> str:
    if runtime_type == RuntimeType.CLOUD:
        lit_app: Externalv1LightningappInstance = args[0]
        action = "?action=add_credits" if need_credits else ""
        return f"{get_lightning_cloud_url()}/me/apps/{lit_app.id}{action}"
    else:
        return "http://127.0.0.1:7501/view"


def main() -> None:
    # Check environment and versions if not in the cloud
    if "LIGHTNING_APP_STATE_URL" not in os.environ:
        # Enforce running in PATH Python
        _check_environment_and_redirect()

        # Check for newer versions and upgrade
        _check_version_and_upgrade()

    # 1: Handle connection to a Lightning App.
    if len(sys.argv) > 1 and sys.argv[1] in ("connect", "disconnect", "logout"):
        _main()
    else:
        # 2: Collect the connection a Lightning App.
        app_name, app_id = _retrieve_connection_to_an_app()
        if app_name:
            # 3: Handle development use case.
            is_local_app = app_name == "localhost"
            if sys.argv[1:3] == ["run", "app"] or (
                sys.argv[1:3] == ["show", "logs"] and "show logs" not in _list_app_commands(False)
            ):
                _main()
            else:
                if is_local_app:
                    message = "You are connected to the local Lightning App."
                else:
                    message = f"You are connected to the cloud Lightning App: {app_name}."

                click.echo(" ")

                if (len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]) or len(sys.argv) == 1:
                    _list_app_commands()
                else:
                    _run_app_command(app_name, app_id)

                click.echo()
                click.echo(message + " Return to the primary CLI with `lightning disconnect`.")
        else:
            _main()


@click.group(cls=_ApiExceptionHandler)
@click.version_option(ver)
def _main() -> None:
    pass


@_main.group()
def show() -> None:
    """Show given resource."""
    pass


_main.command()(connect)
_main.command()(disconnect)
show.command()(logs)


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
    run_app_comment_commands: bool,
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
        if ENABLE_APP_COMMENT_COMMAND_EXECUTION or run_app_comment_commands:
            if file is not None:
                run_app_commands(str(file))

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
        run_app_comment_commands=run_app_comment_commands,
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
@click.option(
    "--setup",
    "-s",
    "run_app_comment_commands",
    is_flag=True,
    default=False,
    help="run environment setup commands from the app comments.",
)
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
    run_app_comment_commands: bool,
) -> None:
    """Run an app from a file."""
    _run_app(
        file,
        cloud,
        cluster_id,
        without_server,
        no_cache,
        name,
        blocking,
        open_ui,
        env,
        secret,
        run_app_comment_commands,
    )


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
_main.add_command(cli_add)
_main.add_command(cli_remove)


@_main.command("ssh")
@click.option(
    "--app-name",
    "app_name",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--component-name",
    "component_name",
    type=str,
    default=None,
    help="Specify which component to SSH into",
)
def ssh(app_name: str = None, component_name: str = None) -> None:
    """SSH into a Lightning App."""

    app_manager = _AppManager()
    apps = app_manager.list_apps(phase_in=[V1LightningappInstanceState.RUNNING])
    if len(apps) == 0:
        raise click.ClickException("No running apps available. Start a Lightning App in the cloud to use this feature.")

    available_app_names = [app.name for app in apps]
    if app_name is None:
        available_apps = [
            inquirer.List(
                "app_name",
                message="What app to SSH into?",
                choices=available_app_names,
            ),
        ]
        app_name = inquirer.prompt(available_apps)["app_name"]
    app_id = next((app.id for app in apps if app.name == app_name), None)
    if app_id is None:
        raise click.ClickException(
            f"Unable to find a running app with name {app_name} in your account. "
            + f"Available running apps are: {', '.join(available_app_names)}"
        )
    try:
        instance = app_manager.get_app(app_id=app_id)
    except ApiException:
        raise click.ClickException("failed fetching app instance")

    components = app_manager.list_components(app_id=app_id)
    available_component_names = [work.name for work in components] + ["flow"]
    if component_name is None:
        available_components = [
            inquirer.List(
                "component_name",
                message="Which component to SSH into?",
                choices=available_component_names,
            )
        ]
        component_name = inquirer.prompt(available_components)["component_name"]

    component_id = None
    if component_name == "flow":
        component_id = f"lightningapp-{app_id}"
    elif component_name is not None:
        work_id = next((work.id for work in components if work.name == component_name), None)
        if work_id is not None:
            component_id = f"lightningwork-{work_id}"

    if component_id is None:
        raise click.ClickException(
            f"Unable to find an app component with name {component_name}. "
            f"Available components are: {', '.join(available_component_names)}"
        )

    app_cluster = app_manager.get_cluster(cluster_id=instance.spec.cluster_id)
    ssh_endpoint = app_cluster.status.ssh_gateway_endpoint

    ssh_path = shutil.which("ssh")
    if ssh_path is None:
        raise click.ClickException(
            "Unable to find the ssh binary. You must install ssh first to use this functionality."
        )
    os.execv(ssh_path, ["-tt", f"{component_id}@{ssh_endpoint}"])


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
