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

import logging
from typing import Tuple

import click

from lightning.app.core.constants import APP_SERVER_HOST, APP_SERVER_PORT
from lightning.app.launcher.launcher import (
    run_lightning_flow,
    run_lightning_work,
    serve_frontend,
    start_application_server,
    start_flow_and_servers,
)

logger = logging.getLogger(__name__)


@click.group(name="launch", hidden=True)
def launch() -> None:
    """Launch your application."""


@launch.command("server", hidden=True)
@click.argument("file", type=click.Path(exists=True))
@click.option("--queue-id", help="ID for identifying queue", default="", type=str)
@click.option("--host", help="Application running host", default=APP_SERVER_HOST, type=str)
@click.option("--port", help="Application running port", default=APP_SERVER_PORT, type=int)
def run_server(file: str, queue_id: str, host: str, port: int) -> None:
    """It takes the application file as input, build the application object and then use that to run the application
    server.

    This is used by the cloud runners to start the status server for the application

    """
    logger.debug(f"Run Server: {file} {queue_id} {host} {port}")
    start_application_server(file, host, port, queue_id=queue_id)


@launch.command("flow", hidden=True)
@click.argument("file", type=click.Path(exists=True))
@click.option("--queue-id", help="ID for identifying queue", default="", type=str)
@click.option("--base-url", help="Base url at which the app server is hosted", default="")
def run_flow(file: str, queue_id: str, base_url: str) -> None:
    """It takes the application file as input, build the application object, proxy all the work components and then run
    the application flow defined in the root component.

    It does exactly what a singleprocess dispatcher would do but with proxied work components.

    """
    logger.debug(f"Run Flow: {file} {queue_id} {base_url}")
    run_lightning_flow(file, queue_id=queue_id, base_url=base_url)


@launch.command("work", hidden=True)
@click.argument("file", type=click.Path(exists=True))
@click.option("--work-name", type=str)
@click.option("--queue-id", help="ID for identifying queue", default="", type=str)
def run_work(file: str, work_name: str, queue_id: str) -> None:
    """Unlike other entrypoints, this command will take the file path or module details for a work component and run
    that by fetching the states from the queues."""
    logger.debug(f"Run Work: {file} {work_name} {queue_id}")
    run_lightning_work(
        file=file,
        work_name=work_name,
        queue_id=queue_id,
    )


@launch.command("frontend", hidden=True)
@click.argument("file", type=click.Path(exists=True))
@click.option("--flow-name")
@click.option("--host")
@click.option("--port", type=int)
def run_frontend(file: str, flow_name: str, host: str, port: int) -> None:
    """Serve the frontend specified by the given flow."""
    logger.debug(f"Run Frontend: {file} {flow_name} {host}")
    serve_frontend(file=file, flow_name=flow_name, host=host, port=port)


@launch.command("flow-and-servers", hidden=True)
@click.argument("file", type=click.Path(exists=True))
@click.option("--queue-id", help="ID for identifying queue", default="", type=str)
@click.option("--base-url", help="Base url at which the app server is hosted", default="")
@click.option("--host", help="Application running host", default=APP_SERVER_HOST, type=str)
@click.option("--port", help="Application running port", default=APP_SERVER_PORT, type=int)
@click.option(
    "--flow-port",
    help="Pair of flow name and frontend port",
    type=(str, int),
    multiple=True,
)
def run_flow_and_servers(
    file: str,
    base_url: str,
    queue_id: str,
    host: str,
    port: int,
    flow_port: Tuple[Tuple[str, int]],
) -> None:
    """It takes the application file as input, build the application object and then use that to run the application
    flow defined in the root component, the application server and all the flow frontends.

    This is used by the cloud runners to start the flow, the status server and all frontends for the application

    """
    logger.debug(f"Run Flow: {file} {queue_id} {base_url}")
    logger.debug(f"Run Server: {file} {queue_id} {host} {port}.")
    logger.debug(f"Run Frontend's: {flow_port}")
    start_flow_and_servers(
        entrypoint_file=file,
        base_url=base_url,
        queue_id=queue_id,
        host=host,
        port=port,
        flow_names_and_ports=flow_port,
    )
