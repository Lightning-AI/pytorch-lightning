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
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from lightning_app.core.constants import get_lightning_cloud_url
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cli_helpers import _error_and_exit

logger = Logger(__name__)

NETWORK_NAME = "lightning-byom"
CMD_CREATE_NETWORK = f"docker network create {NETWORK_NAME}"

CODE_SERVER_CONTAINER = "code-server"
CODE_SERVER_IMAGE = "ghcr.io/gridai/lightning-byom-code-server:v0.1"
CODE_SERVER_PORT = 8443

LIGHTNING_CLOUD_URL = get_lightning_cloud_url()
ROOT_DOMAIN = LIGHTNING_CLOUD_URL.split("//")[1]
CLOUD_PROXY_HOST = f"byom.{ROOT_DOMAIN}"

LIGHTNING_DAEMON_NODE_PREFIX = ""
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


def get_lightning_daemon_command(node_prefix: str) -> str:
    return (
        f"docker run "
        f"-e LIGHTNING_BYOM_CLOUD_PROXY_HOST=https://{node_prefix}.{CLOUD_PROXY_HOST} "
        f"-e LIGHTNING_BYOM_RESOURCE_URL=http://{CODE_SERVER_CONTAINER}:{CODE_SERVER_PORT} "
        f"--net {NETWORK_NAME} "
        f"--name {LIGHTNING_DAEMON_CONTAINER} "
        f"--rm {LIGHTNING_DAEMON_IMAGE}"
    )


@click.argument("name", required=True)
def connect_node(name: str) -> None:
    """Create a new node connection."""
    # print system architecture and OS
    if sys.platform != "darwin" or platform.processor() != "arm":
        _error_and_exit("Node connection is only supported from M1 Macs at the moment")

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
        _error_and_exit("Node connection isn't publicly available. Open an issue on Github.")

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:
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
                live.update(Spinner("point", text=Text("pulling code server image", style="white")))
                out = subprocess.run(f"docker pull {CODE_SERVER_IMAGE}", shell=True, check=True, capture_output=True)
                error = out.stderr
                if error:
                    live.stop()
                    rich.print(f"[red]Failed[/red]: code server image pull failed with error: {str(error)}")
                    return
            cmd = get_code_server_docker_command()
            live.update(Spinner("point", text=Text("running code server", style="white")))
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
                live.update(Spinner("point", text=Text("pulling lightning daemon image", style="white")))
                out = subprocess.run(
                    f"docker pull {LIGHTNING_DAEMON_IMAGE}", shell=True, check=True, capture_output=True
                )
                error = out.stderr
                if error:
                    live.stop()
                    rich.print(f"[red]Failed[/red]: lightnign daemon image pull failed with error: {str(error)}")
                    return
            cmd = get_lightning_daemon_command(name)
            live.update(Spinner("point", text=Text("running lightning daemon", style="white")))
            _ = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # wait until if both docker containers are running
        code_server_running = False
        lightning_daemon_running = False
        live.update(Spinner("point", text=Text("establishing connection ...", style="white")))
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
    rich.print(
        f"[green]Succeeded[/green]: node {name} has been connected to lightning. \n "
        f"Go to https://{name}.{CLOUD_PROXY_HOST} to access the node."
    )


@click.argument("name", required=True)
def disconnect_node(name: str) -> None:
    # disconnect node stop and remove the docker containers
    with Live(Spinner("point", text=Text("disconnecting node...", style="white")), transient=True):
        subprocess.run(f"docker stop {CODE_SERVER_CONTAINER}", shell=True, capture_output=True)
        subprocess.run(f"docker stop {LIGHTNING_DAEMON_CONTAINER}", shell=True, capture_output=True)
    rich.print(f"[green]Succeeded[/green]: node {name} has been disconnected from lightning.")
