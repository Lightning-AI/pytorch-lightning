from lightning_app.cli.cmd_clusters import AWSClusterManager
from lightning_app.cli.lightning_cli import _main


@_main.group(name="list")
def get_list():
    """List your Lightning AI BYOC managed resources."""
    pass


@get_list.command("clusters")
def list_clusters(**kwargs):
    """List your Lightning AI BYOC compute clusters."""
    cluster_manager = AWSClusterManager()
    cluster_manager.list()
