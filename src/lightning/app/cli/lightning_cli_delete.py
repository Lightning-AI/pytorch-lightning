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

import click
import inquirer
from inquirer.themes import GreenPassion
from rich.console import Console

from lightning.app.cli.cmd_apps import _AppManager


@click.group("delete")
def delete() -> None:
    """Delete Lightning AI self-managed resources (e.g. apps)"""
    pass


def _find_selected_app_instance_id(app_name: str) -> str:
    console = Console()
    app_manager = _AppManager()

    all_app_names_and_ids = {}
    selected_app_instance_id = None

    for app in app_manager.list_apps():
        all_app_names_and_ids[app.name] = app.id
        # figure out the ID of some app_name
        if app_name == app.name or app_name == app.id:
            selected_app_instance_id = app.id
            break

    if selected_app_instance_id is None:
        # when there is no app with the given app_name,
        # ask the user which app they would like to delete.
        console.print(f'[b][yellow]Cannot find app named "{app_name}"[/yellow][/b]')
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


def _delete_app_confirmation_prompt(app_name: str) -> None:
    console = Console()

    # when the --yes / -y flags were not passed, do a final
    # confirmation that the user wants to delete the app.
    try:
        ask = [
            inquirer.Confirm(
                "confirm",
                message=f'Are you sure you want to delete app "{app_name}""?',
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
    "skip_user_confirm_prompt",
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Do not prompt for confirmation.",
)
def delete_app(app_name: str, skip_user_confirm_prompt: bool) -> None:
    """Delete a Lightning app.

    Deleting an app also deletes all app websites, works, artifacts, and logs. This permanently removes any record of
    the app as well as all any of its associated resources and data. This does not affect any resources and data
    associated with other Lightning apps on your account.

    """
    console = Console()

    try:
        selected_app_instance_id = _find_selected_app_instance_id(app_name=app_name)
        if not skip_user_confirm_prompt:
            _delete_app_confirmation_prompt(app_name=app_name)
    except InterruptedError:
        return

    try:
        # Delete the app!
        app_manager = _AppManager()
        app_manager.delete(app_id=selected_app_instance_id)
    except Exception as ex:
        console.print(
            f'[b][red]An issue occurred while deleting app "{app_name}. If the issue persists, please '
            "reach out to us at [link=mailto:support@lightning.ai]support@lightning.ai[/link][/b][/red]."
        )
        raise click.ClickException(str(ex))

    console.print(f'[b][green]App "{app_name}" has been successfully deleted"![/green][/b]')
    return
