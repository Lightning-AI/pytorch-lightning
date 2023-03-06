# Copyright The Lightning team.
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
import platform
import shlex
import subprocess
import sys
import time

import click
import rich
from lightning_cloud.openapi import (
    V1BYOMClusterDriver,
    V1ClusterDriver,
    V1ClusterSpec,
    V1ClusterState,
    V1ClusterType,
    V1CreateClusterRequest,
)
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.core.constants import get_lightning_cloud_url
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cli_helpers import _error_and_exit
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.clusters import _ensure_cluster_project_binding
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)

NETWORK_NAME = "lightning-maverick"
CMD_CREATE_NETWORK = f"docker network create {NETWORK_NAME}"

CODE_SERVER_CONTAINER = "code-server"
CODE_SERVER_IMAGE = "ghcr.io/gridai/lightning-maverick-code-server:v0.1"
CODE_SERVER_PORT = 8443

LIGHTNING_CLOUD_URL = get_lightning_cloud_url()
ROOT_DOMAIN = LIGHTNING_CLOUD_URL.split("//")[1]
CLOUD_PROXY_HOST = f"byom.{ROOT_DOMAIN}"

LIGHTNING_DAEMON_CONTAINER = "lightning-daemon"
LIGHTNING_DAEMON_IMAGE = "ghcr.io/gridai/lightning-daemon:v0.1"


def get_code_server_docker_command() -> str:
    return (
        f"docker run "
        f"-p {CODE_SERVER_PORT}:{CODE_SERVER_PORT} "
        f"--net {NETWORK_NAME} "
        f"--name {CODE_SERVER_CONTAINER} "
        f"--rm {CODE_SERVER_IMAGE}"
    )


def get_lightning_daemon_command(prefix: str) -> str:
    return (
        f"docker run "
        f"-e LIGHTNING_BYOM_CLOUD_PROXY_HOST=https://{prefix}.{CLOUD_PROXY_HOST} "
        f"-e LIGHTNING_BYOM_RESOURCE_URL=http://{CODE_SERVER_CONTAINER}:{CODE_SERVER_PORT} "
        f"--net {NETWORK_NAME} "
        f"--name {LIGHTNING_DAEMON_CONTAINER} "
        f"--rm {LIGHTNING_DAEMON_IMAGE}"
    )


@click.argument("name", required=True)
@click.option("--project_name", help="The project name to which the machine should connect.", required=False)
def connect_maverick(name: str, project_name: str = "") -> None:
    """Create a new maverick connection."""
    # print system architecture and OS
    if sys.platform != "darwin" or platform.processor() != "arm":
        _error_and_exit("Maverick connection is only supported from M1 Macs at the moment")

    # check if docker client is installed or not
    try:
        subprocess.run("docker --version", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        _error_and_exit("Docker client is not installed. Please install docker and try again.")

    # check if docker daemon is running or not
    try:
        subprocess.run("docker ps", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        _error_and_exit("Docker daemon is not running. Please start docker and try again.")

    if "lightning.ai" in CLOUD_PROXY_HOST:
        _error_and_exit("Maverick connection isn't publicly available. Open an issue on Github.")

    with Live(Spinner("point", text=Text("Registering maverick...", style="white")), transient=True) as live:
        try:
            register_to_cloud(name, project_name)
        except Exception as e:
            live.stop()
            rich.print(f"[red]Failed[/red]: Registering maverick failed with error {e}")
            return

        live.update(Spinner("point", text=Text("Setting up ...", style="white")))

        # run network creation in the background
        out = subprocess.run(CMD_CREATE_NETWORK, shell=True, capture_output=True)
        error = out.stderr
        if error:
            if "already exists" not in str(error):
                live.stop()
                rich.print(f"[red]Failed[/red]: network creation failed with error: {str(error)}")
                return

        # if code server is already running, ignore.
        # If not, but container exists, remove it and run. Otherwise, run.
        out = subprocess.run(
            f"docker ps -q -f name={CODE_SERVER_CONTAINER}", shell=True, check=True, capture_output=True
        )
        if out.stdout:
            pass
        else:
            out = subprocess.run(
                f"docker container ls -aq -f name={CODE_SERVER_CONTAINER}", shell=True, check=True, capture_output=True
            )
            if out.stdout:
                subprocess.run(f"docker rm -f {CODE_SERVER_CONTAINER}", shell=True, check=True)
            else:
                out = subprocess.run(f"docker pull {CODE_SERVER_IMAGE}", shell=True, check=True, capture_output=True)
                error = out.stderr
                if error:
                    live.stop()
                    rich.print(f"[red]Failed[/red]: code server image pull failed with error: {str(error)}")
                    return
            cmd = get_code_server_docker_command()
            _ = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # if lightning daemon is already running, ignore.
        # If not, but container exists, remove it and run. Otherwise, run.
        out = subprocess.run(
            f"docker ps -q -f name={LIGHTNING_DAEMON_CONTAINER}", shell=True, check=True, capture_output=True
        )
        if out.stdout:
            pass
        else:
            out = subprocess.run(
                f"docker container ls -aq -f name={LIGHTNING_DAEMON_CONTAINER}",
                shell=True,
                check=True,
                capture_output=True,
            )
            if out.stdout:
                subprocess.run(f"docker rm -f {LIGHTNING_DAEMON_CONTAINER}", shell=True, check=True)
            else:
                out = subprocess.run(
                    f"docker pull {LIGHTNING_DAEMON_IMAGE}", shell=True, check=True, capture_output=True
                )
                error = out.stderr
                if error:
                    live.stop()
                    rich.print(f"[red]Failed[/red]: lightnign daemon image pull failed with error: {str(error)}")
                    return
            cmd = get_lightning_daemon_command(name)
            _ = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # wait until if both docker containers are running
        code_server_running = False
        lightning_daemon_running = False
        live.update(Spinner("point", text=Text("Establishing connection ...", style="white")))
        connection_check_start_time = time.time()

        # wait for 30 seconds for connection to be established
        while time.time() - connection_check_start_time < 30:
            out = subprocess.run(
                f"docker container ls -f name={CODE_SERVER_CONTAINER} " + '--format "{{.Status}}"',
                shell=True,
                check=True,
                capture_output=True,
            )

            if "Up" in str(out.stdout):
                code_server_running = True

            out = subprocess.run(
                f"docker container ls -f name={LIGHTNING_DAEMON_CONTAINER} " + '--format "{{.Status}}"',
                shell=True,
                check=True,
                capture_output=True,
            )
            if "Up" in str(out.stdout):
                lightning_daemon_running = True

            if code_server_running and lightning_daemon_running:
                break

            # Sleeping for 0.5 seconds
            time.sleep(0.5)
    rich.print(f"[green]Succeeded[/green]: maverick {name} has been connected to lightning.")


@click.argument("name", required=True)
def disconnect_maverick(name: str) -> None:
    # disconnect stop and remove the docker containers
    with Live(Spinner("point", text=Text("disconnecting maverick...", style="white")), transient=True):
        try:
            deregister_from_cloud(name)
        except Exception as e:
            rich.print(f"[red]Failed[/red]: Disconnecting machine failed with error: {e}")
            return
        subprocess.run(f"docker stop {CODE_SERVER_CONTAINER}", shell=True, capture_output=True)
        subprocess.run(f"docker stop {LIGHTNING_DAEMON_CONTAINER}", shell=True, capture_output=True)
    rich.print(f"[green]Succeeded[/green]: maverick {name} has been disconnected from lightning.")


def register_to_cloud(name: str, project_name: str) -> None:
    client = LightningClient(retry=False)
    projects = client.projects_service_list_memberships()
    if project_name:
        for project in projects.memberships:
            if project.name == project_name:
                project_id = project.project_id
                break
        else:
            raise ValueError(f"Project {project_name} does not exist.")
    else:
        project_id = _get_project(client, verbose=False).project_id

    cluster_bindings = client.projects_service_list_project_cluster_bindings(project_id=project_id)
    for c in cluster_bindings.clusters:
        if c.cluster_name == name:
            existing_cluster = client.cluster_service_get_cluster(id=c.cluster_id)
            if existing_cluster.status.phase == V1ClusterState.RUNNING:
                raise RuntimeError(f"Cluster {name} already exists and is running.")
            break
    else:
        body = V1CreateClusterRequest(
            name=name,
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.BYOM,
                driver=V1ClusterDriver(byom=V1BYOMClusterDriver()),
            ),
        )
        resp = client.cluster_service_create_cluster(body=body)
        _ensure_cluster_project_binding(client, project_id, resp.id)


def deregister_from_cloud(name: str) -> None:
    client = LightningClient(retry=False)
    clusters = client.cluster_service_list_clusters()
    # TODO (sherin) this should wait for gridlet to stop running before deleting the cluster
    for cluster in clusters.clusters:
        if cluster.name == name:
            client.cluster_service_delete_cluster(id=cluster.id, force=True)
            break
    else:
        raise RuntimeError(f"Cluster {name} does not exist.")
