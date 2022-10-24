import click


@click.group("remove")
def cli_remove() -> None:
    """Remove Lightning AI self-managed resources (ssh-keys, etc…)"""
    pass


@cli_remove.command("ssh-key")
def remove_ssh_key() -> None:
    pass
