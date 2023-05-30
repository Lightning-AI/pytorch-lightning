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

from typing import Any

import click

from lightning.app.cli.cmd_apps import _AppManager
from lightning.app.cli.cmd_clusters import AWSClusterManager
from lightning.app.cli.cmd_ssh_keys import _SSHKeyManager


@click.group(name="list")
def get_list() -> None:
    """List Lightning AI self-managed resources (clusters, etc…)"""
    pass


@get_list.command("clusters")
def list_clusters(**kwargs: Any) -> None:
    """List your Lightning AI BYOC compute clusters."""
    cluster_manager = AWSClusterManager()
    cluster_manager.list()


@click.option(
    "--cluster-id",
    "cluster_id",
    type=str,
    required=False,
    default=None,
    help="Filter apps by associated Lightning AI compute cluster",
)
@get_list.command("apps")
def list_apps(cluster_id: str, **kwargs: Any) -> None:
    """List your Lightning AI apps."""
    app_manager = _AppManager()
    app_manager.list(cluster_id=cluster_id)


@get_list.command("ssh-keys")
def list_ssh_keys() -> None:
    """List your Lightning AI ssh-keys."""
    ssh_key_manager = _SSHKeyManager()
    ssh_key_manager.list()
