import click

from lightning_app.cli.cmd_ssh_keys import SSHKeyManager


@click.group("remove")
def cli_remove() -> None:
    """Remove Lightning AI self-managed resources (ssh-keys, etc…)"""
    pass


@cli_remove.command("ssh-key")
@click.argument("key_id")
def remove_ssh_key(key_id: str) -> None:
    """Remove a ssh-key from your Lightning AI account."""
    ssh_key_manager = SSHKeyManager()
    ssh_key_manager.remove_key(key_id=key_id)
