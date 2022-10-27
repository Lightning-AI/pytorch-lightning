import os
from pathlib import Path
from typing import Optional, Union

import click
from lightning_cloud.openapi.rest import ApiException

from lightning_app.cli.cmd_ssh_keys import _SSHKeyManager


@click.group("add")
def cli_add() -> None:
    """Add Lightning AI self-managed resources (ssh-keys, etc…)"""
    pass


@cli_add.command("ssh-key")
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
