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

from typing import List

import click
import rich
from rich.color import ANSI_COLOR_NAMES

from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.app_logs import _app_logs_reader
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.logs_socket_api import _LightningLogsSocketAPI
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("app_name", required=False)
@click.argument("components", nargs=-1, required=False)
@click.option("-f", "--follow", required=False, is_flag=True, help="Wait for new logs, to exit use CTRL+C.")
def logs(app_name: str, components: List[str], follow: bool) -> None:
    """Show cloud application logs. By default, prints logs for all currently available components.

    Example uses:

        Print all application logs:

            $ lightning show logs my-application


        Print logs only from the flow (no work):

            $ lightning show logs my-application flow


        Print logs only from selected works:

            $ lightning show logs my-application root.work_a root.work_b
    """
    _show_logs(app_name, components, follow)


def _show_logs(app_name: str, components: List[str], follow: bool) -> None:
    client = LightningClient(retry=False)
    project = _get_project(client)

    apps = {
        getattr(app, "display_name", None) or app.name: app
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
            f"You have not specified any Lightning App. Please select one of the following: [{', '.join(apps.keys())}]."
        )

    if app_name not in apps:
        raise click.ClickException(
            f"The Lightning App '{app_name}' does not exist. "
            f"Please select one of the following: [{', '.join(apps.keys())}]."
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
