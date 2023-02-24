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
from pathlib import Path
from typing import Any, Optional, Union

import click
from lightning_cloud.openapi.rest import ApiException

from lightning.app.cli.cmd_clusters import _check_cluster_id_is_valid, AWSClusterManager
from lightning.app.cli.cmd_ssh_keys import _SSHKeyManager


@click.group("create")
def create() -> None:
    """Create Lightning AI self-managed resources (clusters, etc…)"""
    pass


@create.command("cluster")
@click.argument("cluster_id", callback=_check_cluster_id_is_valid)
@click.option("--provider", "provider", type=str, default="aws", help="cloud provider to be used for your cluster")
@click.option("--external-id", "external_id", type=str, required=True)
@click.option(
    "--role-arn", "role_arn", type=str, required=True, help="AWS role ARN attached to the associated resources."
)
@click.option(
    "--region",
    "region",
    type=str,
    required=False,
    default="us-east-1",
    help="AWS region that is used to host the associated resources.",
    hidden=True,
)
@click.option(
    "--enable-performance",
    "enable_performance",
    type=bool,
    required=False,
    default=False,
    hidden=True,
    is_flag=True,
    help=""""Use this flag to ensure that the cluster is created with a profile that is optimized for performance.
        This makes runs more expensive but start-up times decrease.""",
)
@click.option(
    "--edit-before-creation",
    default=False,
    is_flag=True,
    hidden=True,
    help="Edit the cluster specs before submitting them to the API server.",
)
@click.option(
    "--sync",
    "do_sync",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="This flag makes the CLI wait until cluster creation completes.",
)
def create_cluster(
    cluster_id: str,
    region: str,
    role_arn: str,
    external_id: str,
    provider: str,
    edit_before_creation: bool,
    enable_performance: bool,
    do_sync: bool,
    **kwargs: Any,
) -> None:
    """Create a Lightning AI BYOC compute cluster with your cloud provider credentials."""
    if provider.lower() != "aws":
        click.echo("Only AWS is supported for now. But support for more providers is coming soon.")
        return
    cluster_manager = AWSClusterManager()
    cluster_manager.create(
        cluster_id=cluster_id,
        region=region,
        role_arn=role_arn,
        external_id=external_id,
        edit_before_creation=edit_before_creation,
        cost_savings=not enable_performance,
        do_async=not do_sync,
    )


@create.command("ssh-key")
@click.option("--name", "key_name", default=None, help="name of ssh key")
@click.option("--comment", "comment", default="", help="comment detailing your SSH key")
@click.option(
    "--public-key",
    "public_key",
    help="public key or path to public key file",
    required=True,
)
def add_ssh_key(
    public_key: Union[str, "os.PathLike[str]"], key_name: Optional[str] = None, comment: Optional[str] = None
) -> None:
    """Add a new Lightning AI ssh-key to your account."""
    ssh_key_manager = _SSHKeyManager()

    new_public_key = Path(str(public_key)).read_text() if os.path.isfile(str(public_key)) else public_key
    try:
        ssh_key_manager.add_key(name=key_name, comment=comment, public_key=str(new_public_key))
    except ApiException as e:
        # if we got an exception it might be the user passed the private key file
        if os.path.isfile(str(public_key)) and os.path.isfile(f"{public_key}.pub"):
            ssh_key_manager.add_key(name=key_name, comment=comment, public_key=Path(f"{public_key}.pub").read_text())
        else:
            raise e
