import click

from lightning_app.cli.cmd_clusters import AWSClusterManager


@click.group(name="list")
def get_list():
    """List your Lightning AI BYOC managed resources."""
    pass


@get_list.command("clusters")
def list_clusters(**kwargs):
    """List your Lightning AI BYOC compute clusters."""
    cluster_manager = AWSClusterManager()
    cluster_manager.list()
