import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click
import requests
from requests.exceptions import ConnectionError

from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.core.constants import get_lightning_cloud_url, LOCAL_LAUNCH_ADMIN_VIEW
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.utilities.cli_helpers import _format_input_env_variables
from lightning_app.utilities.install_components import register_all_external_components
from lightning_app.utilities.login import Auth
from lightning_app.utilities.state import headers_for

logger = logging.getLogger(__name__)


def get_app_url(runtime_type: RuntimeType, *args) -> str:
    if runtime_type == RuntimeType.CLOUD:
        lightning_app = args[0]
        return f"{get_lightning_cloud_url()}/me/apps/{lightning_app.id}"
    else:
        return "http://127.0.0.1:7501/admin" if LOCAL_LAUNCH_ADMIN_VIEW else "http://127.0.0.1:7501/view"


@click.group()
@click.version_option(ver)
def main():
    register_all_external_components()
    pass


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
def run_app(
    file: str, cloud: bool, without_server: bool, no_cache: bool, name: str, blocking: bool, open_ui: bool, env: tuple
):
    """Run an app from a file."""
    _run_app(file, cloud, without_server, no_cache, name, blocking, open_ui, env)


@main.group()
def exec():
    """exec your application."""


def _retrieve_application_url(app_id_or_name: Optional[str]):
    failed_locally = False

    if app_id_or_name is None:
        try:
            url = "http://127.0.0.1:7501"
            response = requests.get(f"{url}/api/v1/commands")
            assert response.status_code == 200
            return url, response.json()
        except ConnectionError:
            failed_locally = True

    if app_id_or_name or failed_locally:
        from lightning_app.utilities.cloud import _get_project
        from lightning_app.utilities.network import LightningClient

        client = LightningClient()
        project = _get_project(client)
        list_lightningapps = client.lightningapp_instance_service_list_lightningapp_instances(project.project_id)

        lightningapp_names = [lightningapp.name for lightningapp in list_lightningapps.lightningapps]

        if not app_id_or_name:
            raise Exception(f"Provide an application name or id with --app_id_or_name=X. Found {lightningapp_names}")

        for lightningapp in list_lightningapps.lightningapps:
            if lightningapp.id == app_id_or_name or lightningapp.name == app_id_or_name:
                response = requests.get(lightningapp.status.url + "/api/v1/commands")
                assert response.status_code == 200
                return lightningapp.status.url, response.json()
    return None, None


@exec.command("app")
@click.argument("command", type=str, default="")
@click.option("--args", type=str, default=[], multiple=True, help="Env variables to be set for the app.")
@click.option("--app_id_or_name", help="The current application name", default="", type=str)
def exec_app(
    command: str,
    args: List[str],
    app_id_or_name: Optional[str] = None,
):
    """Run an app from a file."""
    url, commands = _retrieve_application_url(app_id_or_name)
    if url is None or commands is None:
        raise Exception("We couldn't find any matching running app.")

    if not commands:
        raise Exception("This application doesn't expose any commands yet.")

    command_names = [c["command"] for c in commands]
    if command not in command_names:
        raise Exception(f"The provided command {command} isn't available in {command_names}")

    command_metadata = [c for c in commands if c["command"] == command][0]
    params = command_metadata["params"]
    kwargs = {k.split("=")[0]: k.split("=")[1] for k in args}
    for param in params:
        if param not in kwargs:
            raise Exception(f"The argument --args {param}=X hasn't been provided.")

    json = {
        "command_name": command,
        "command_arguments": kwargs,
        "affiliation": command_metadata["affiliation"],
    }
    response = requests.post(url + "/api/v1/commands", json=json, headers=headers_for({}))
    assert response.status_code == 200, response.json()


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
