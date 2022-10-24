import click


@click.group("add")
def cli_add() -> None:
    """Add Lightning AI self-managed resources (ssh-keys, etc…)"""
    pass


@cli_add.command("ssh-key")
def add_ssh_key() -> None:
    pass
