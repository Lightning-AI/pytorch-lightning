import os
from pathlib import Path
from typing import Union

import click

from lightning_app.cli.cmd_ssh_keys import SSHKeyManager


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
def add_ssh_key(key_name: str, comment: str, public_key: Union[str, "os.PathLike[str]"] = None) -> None:
    """Add a new Lightning AI ssh-key to your account."""
    ssh_key_manager = SSHKeyManager()

    new_public_key = Path(public_key).read_text() if os.path.isfile(public_key) else public_key
    ssh_key_manager.add_key(name=key_name, comment=comment, public_key=new_public_key)
