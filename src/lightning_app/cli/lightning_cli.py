import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Union
from uuid import uuid4

import click
import requests
import rich
from requests.exceptions import ConnectionError
from rich.color import ANSI_COLOR_NAMES

from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.cli.lightning_cli_create import create
from lightning_app.cli.lightning_cli_delete import delete
from lightning_app.cli.lightning_cli_list import get_list
from lightning_app.core.constants import get_lightning_cloud_url, LOCAL_LAUNCH_ADMIN_VIEW
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.utilities.app_logs import _app_logs_reader
from lightning_app.utilities.cli_helpers import (
    _format_input_env_variables,
    _retrieve_application_url_and_available_commands,
)
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.install_components import register_all_external_components
from lightning_app.utilities.login import Auth
from lightning_app.utilities.network import LightningClient
from lightning_app.utilities.state import headers_for

logger = logging.getLogger(__name__)


def get_app_url(runtime_type: RuntimeType, *args) -> str:
    if runtime_type == RuntimeType.CLOUD:
        lightning_app = args[0]
        return f"{get_lightning_cloud_url()}/me/apps/{lightning_app.id}"
    else:
        return "http://127.0.0.1:7501/admin" if LOCAL_LAUNCH_ADMIN_VIEW else "http://127.0.0.1:7501/view"


def main():
    if len(sys.argv) == 1:
        _main()
    elif sys.argv[1] in _main.commands.keys() or sys.argv[1] == "--help":
        _main()
    else:
        app_command()


@click.group()
@click.version_option(ver)
def _main():
    register_all_external_components()


@_main.group()
def show():
    """Show given resource."""
    pass


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
        for app in client.lightningapp_instance_service_list_lightningapp_instances(project.project_id).lightningapps
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

    for component in components:
        if component not in app_component_names:
            raise click.ClickException(f"Component '{component}' does not exist in app {app_name}.")

    log_reader = _app_logs_reader(
        client=client,
        project_id=project.project_id,
        app_id=apps[app_name].id,
        component_names=components,
        follow=follow,
    )

    rich_colors = list(ANSI_COLOR_NAMES)
    colors = {c: rich_colors[i + 1] for i, c in enumerate(components)}

    for component_name, log_event in log_reader:
        date = log_event.timestamp.strftime("%m/%d/%Y %H:%M:%S")
        color = colors[component_name]
        rich.print(f"[{color}]{component_name}[/{color}] {date} {log_event.message}")


@_main.command()
def login():
    """Log in to your Lightning.ai account."""
    auth = Auth()
    auth.clear()

    try:
        auth._run_server()
    except ConnectionError:
        click.echo(f"Unable to connect to {get_lightning_cloud_url()}. Please check your internet connection.")
        exit(1)


@_main.command()
def logout():
    """Log out of your Lightning.ai account."""
    Auth().clear()


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
):
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
        cluster_id=cluster_id,
    )
    if runtime_type == RuntimeType.CLOUD:
        click.echo("Application is ready in the cloud")


@_main.group()
def run():
    """Run your application."""


@run.command("app")
@click.argument("file", type=click.Path(exists=True))
@click.option("--cloud", type=bool, default=False, is_flag=True)
@click.option(
    "--cluster-id", type=str, default=None, help="Run Lightning App on a specific Lightning AI BYOC compute cluster"
)
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
    cluster_id: str,
    without_server: bool,
    no_cache: bool,
    name: str,
    blocking: bool,
    open_ui: bool,
    env: tuple,
    app_args: List[str],
):
    """Run an app from a file."""
    _run_app(file, cloud, cluster_id, without_server, no_cache, name, blocking, open_ui, env)


def app_command():
    """Execute a function in a running application from its name."""
    from lightning_app.utilities.commands.base import _download_command

    logger.warn("Lightning Commands are a beta feature and APIs aren't stable yet.")

    debug_mode = bool(int(os.getenv("DEBUG", "0")))

    parser = ArgumentParser()
    parser.add_argument("--app_id", default=None, type=str, help="Optional argument to identify an application.")
    hparams, argv = parser.parse_known_args()

    # 1: Collect the url and comments from the running application
    url, commands = _retrieve_application_url_and_available_commands(hparams.app_id)
    if url is None or commands is None:
        raise Exception("We couldn't find any matching running app.")

    if not commands:
        raise Exception("This application doesn't expose any commands yet.")

    command = argv[0]

    command_names = [c["command"] for c in commands]
    if command not in command_names:
        raise Exception(f"The provided command {command} isn't available in {command_names}")

    # 2: Send the command from the user
    command_metadata = [c for c in commands if c["command"] == command][0]
    params = command_metadata["params"]

    # 3: Execute the command
    if not command_metadata["is_client_command"]:
        # TODO: Improve what is supported there.
        kwargs = {k.split("=")[0].replace("--", ""): k.split("=")[1] for k in argv[1:]}
        for param in params:
            if param not in kwargs:
                raise Exception(f"The argument --{param}=X hasn't been provided.")
        json = {
            "command_name": command,
            "command_arguments": kwargs,
            "affiliation": command_metadata["affiliation"],
            "id": str(uuid4()),
        }
        resp = requests.post(url + "/api/v1/commands", json=json, headers=headers_for({}))
        assert resp.status_code == 200, resp.json()
    else:
        client_command, models = _download_command(command_metadata, hparams.app_id, debug_mode=debug_mode)
        client_command._setup(metadata=command_metadata, models=models, app_url=url)
        sys.argv = argv
        client_command.run()


@_main.group(hidden=True)
def fork():
    """Fork an application."""
    pass


@_main.group(hidden=True)
def stop():
    """Stop your application."""
    pass


_main.add_command(get_list)
_main.add_command(delete)
_main.add_command(create)


@_main.group()
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


@_main.group()
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
