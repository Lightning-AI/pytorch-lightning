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

import os
import shutil
import sys
from pathlib import Path
from typing import Tuple, Union

import arrow
import click
import inquirer
import rich
from lightning_cloud.openapi import V1LightningappInstanceState, V1LightningworkState
from lightning_cloud.openapi.rest import ApiException
from lightning_utilities.core.imports import RequirementCache
from requests.exceptions import ConnectionError

import lightning_app.core.constants as constants
from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.cli.cmd_apps import _AppManager
from lightning_app.cli.cmd_clusters import AWSClusterManager
from lightning_app.cli.commands.app_commands import _run_app_command
from lightning_app.cli.commands.cd import cd
from lightning_app.cli.commands.cp import cp
from lightning_app.cli.commands.logs import logs
from lightning_app.cli.commands.ls import ls
from lightning_app.cli.commands.pwd import pwd
from lightning_app.cli.commands.rm import rm
from lightning_app.cli.connect.app import (
    _list_app_commands,
    _retrieve_connection_to_an_app,
    connect_app,
    disconnect_app,
)
from lightning_app.cli.connect.data import connect_data
from lightning_app.cli.connect.node import connect_node, disconnect_node
from lightning_app.cli.lightning_cli_create import create
from lightning_app.cli.lightning_cli_delete import delete
from lightning_app.cli.lightning_cli_list import get_list
from lightning_app.core.constants import DEBUG, ENABLE_APP_COMMENT_COMMAND_EXECUTION, get_lightning_cloud_url
from lightning_app.runners.cloud import CloudRuntime
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
from lightning_app.utilities.port import _find_lit_app_port

logger = Logger(__name__)


def main() -> None:
    # Check environment and versions if not in the cloud and not testing
    is_testing = bool(int(os.getenv("LIGHTING_TESTING", "0")))
    if not is_testing and "LIGHTNING_APP_STATE_URL" not in os.environ:
        try:
            # Enforce running in PATH Python
            _check_environment_and_redirect()

            # Check for newer versions and upgrade
            _check_version_and_upgrade()
        except SystemExit:
            raise
        except Exception:
            # Note: We intentionally ignore all exceptions here so that we never panic if one of the above calls fails.
            # If they fail for some reason users should still be able to continue with their command.
            click.echo(
                "We encountered an unexpected problem while checking your environment."
                "We will still proceed with the command, however, there is a chance that errors may occur."
            )

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


@_main.group()
def connect() -> None:
    """Connect apps and data."""
    pass


@_main.group()
def disconnect() -> None:
    """Disconnect apps."""
    pass


connect.command("app")(connect_app)
disconnect.command("app")(disconnect_app)
connect.command("node", hidden=True)(connect_node)
disconnect.command("node", hidden=True)(disconnect_node)
connect.command("data", hidden=True)(connect_data)
_main.command(hidden=True)(ls)
_main.command(hidden=True)(cd)
_main.command(hidden=True)(cp)
_main.command(hidden=True)(pwd)
_main.command(hidden=True)(rm)
show.command()(logs)


@show.group()
def cluster() -> None:
    """Groups cluster commands inside show."""
    pass


@cluster.command(name="logs")
@click.argument("cluster_id", required=True)
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
def cluster_logs(cluster_id: str, to_time: arrow.Arrow, from_time: arrow.Arrow, limit: int, follow: bool) -> None:
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

    client = LightningClient(retry=False)
    cluster_manager = AWSClusterManager()
    existing_cluster_list = cluster_manager.get_clusters()

    clusters = {cluster.id: cluster.id for cluster in existing_cluster_list.clusters}

    if not clusters:
        raise click.ClickException("You don't have any clusters.")

    if not cluster_id:
        raise click.ClickException(
            f"You have not specified any clusters. Please select one of available: [{', '.join(clusters.keys())}]"
        )

    if cluster_id not in clusters:
        raise click.ClickException(
            f"The cluster '{cluster_id}' does not exist."
            f" Please select one of the following: [{', '.join(clusters.keys())}]"
        )

    try:
        log_reader = _cluster_logs_reader(
            logs_api_client=_ClusterLogsSocketAPI(client.api_client),
            cluster_id=cluster_id,
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
        auth.authenticate()
    except ConnectionError:
        click.echo(f"Unable to connect to {get_lightning_cloud_url()}. Please check your internet connection.")
        exit(1)


@_main.command()
def logout() -> None:
    """Log out of your lightning.ai account."""
    Auth().clear()
    disconnect_app(logout=True)


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
    enable_basic_auth: str,
) -> None:

    if not os.path.exists(file):
        original_file = file
        file = cmd_install.gallery_apps_and_components(file, True, "latest", overwrite=True)  # type: ignore[assignment]  # noqa E501
        if file is None:
            click.echo(f"The provided entrypoint `{original_file}` doesn't exist.")
            sys.exit(1)
        run_app_comment_commands = True

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

    port = _find_lit_app_port(constants.APP_SERVER_PORT)
    constants.APP_SERVER_PORT = port

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
        open_ui=open_ui,
        name=name,
        env_vars=env_vars,
        secrets=secrets,
        cluster_id=cluster_id,
        run_app_comment_commands=run_app_comment_commands,
        enable_basic_auth=enable_basic_auth,
        port=port,
    )
    if runtime_type == RuntimeType.CLOUD:
        click.echo("Application is ready in the cloud")


@_main.group()
def run() -> None:
    """Run a Lightning application locally or on the cloud."""


@run.command("app")
@click.argument("file", type=str)
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
@click.option(
    "--enable-basic-auth",
    type=str,
    default="",
    help="Enable basic authentication for the app and use credentials provided in the format username:password",
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
    enable_basic_auth: str,
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
        enable_basic_auth,
    )


if RequirementCache("lightning-fabric>=1.9.0") or RequirementCache("lightning>=1.9.0"):
    # note it is automatically replaced to `from lightning.fabric.cli` when building monolithic/mirror package
    from lightning_fabric.cli import _run_model

    run.add_command(_run_model)


@_main.command("open", hidden=True)
@click.argument("path", type=str, default=".")
@click.option(
    "--cluster-id",
    type=str,
    default=None,
    help="Open on a specific Lightning AI BYOC compute cluster",
)
@click.option("--name", help="The name to use for the CloudSpace", default="", type=str)
def open(path: str, cluster_id: str, name: str) -> None:
    """Open files or folders from your machine in a Lightning CloudSpace."""

    if not os.path.exists(path):
        click.echo(f"The provided path `{path}` doesn't exist.")
        sys.exit(1)

    runtime = CloudRuntime(entrypoint=Path(path))
    runtime.open(name, cluster_id)


_main.add_command(get_list)
_main.add_command(delete)
_main.add_command(create)
_main.add_command(cmd_install.install)


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

    components = app_manager.list_components(app_id=app_id, phase_in=[V1LightningworkState.RUNNING])
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
