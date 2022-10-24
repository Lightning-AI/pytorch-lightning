from pathlib import Path

import click

from lightning_app.cli.cmd_ssh_keys import SSHKeyManager


@click.group("add")
def cli_add() -> None:
    """Add Lightning AI self-managed resources (ssh-keys, etc…)"""
    pass


@cli_add.command("ssh-key")
@click.argument("key_name")
@click.option("--comment", "comment", type=str, default="", help="comment detailing your SSH key")
@click.option(
    "--public-key-path",
    "public_key_path",
    type=click.Path(exists=True),
    default=None,
    help="path to your public key file",
)
@click.option(
    "--public-key",
    "public_key",
    type=str,
    default=None,
    help="public key",
)
def add_ssh_key(key_name: str, comment: str, public_key_path: str = None, public_key: str = None) -> None:
    """Add a new Lightning AI ssh-key to your account."""
    # https://github.com/pallets/click/issues/257 sadly click can't model "1 of N options must be provided"
    if public_key is None and public_key_path is None:
        raise click.ClickException("One of --public-key or --public-key-path must be provided")

    ssh_key_manager = SSHKeyManager()

    new_public_key = public_key if public_key is not None else Path(public_key_path).read_text()
    ssh_key_manager.add_key(name=key_name, comment=comment, public_key=new_public_key)
