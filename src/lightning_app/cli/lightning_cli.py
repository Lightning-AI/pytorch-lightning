import logging
import os
from pathlib import Path
from typing import Tuple, Union

import click
from requests.exceptions import ConnectionError

from lightning_app import __version__ as ver
from lightning_app.cli import cmd_init, cmd_install, cmd_pl_init, cmd_react_ui_init
from lightning_app.core.constants import get_lightning_cloud_url, LOCAL_LAUNCH_ADMIN_VIEW
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.utilities.cli_helpers import _format_input_env_variables
from lightning_app.utilities.install_components import register_all_external_components
from lightning_app.utilities.login import Auth

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

    if not exists and file == "quick_start.py":
        from lightning_app.demo.quick_start import app

        logger.info(f"For demo purposes, Lightning will run the {app.__file__} file.")
        return app.__file__

    raise FileNotFoundError(f"The provided file {file} hasn't been found.")
