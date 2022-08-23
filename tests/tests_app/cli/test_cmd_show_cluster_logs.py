from unittest import mock

from click.testing import CliRunner

from lightning_app.cli.cmd_clusters import ClusterList
from lightning_app.cli.lightning_cli import cluster_logs


@mock.patch("lightning_app.cli.lightning_cli.AWSClusterManager.get_clusters")
def test_show_logs_errors(client, get_clusters):
    """Test that the CLI prints the errors for the show logs command."""

    runner = CliRunner()

    # Run without arguments
    get_clusters.return_value = ClusterList([])
    result = runner.invoke(cluster_logs, [])

    assert result.exit_code == 2
    assert "Usage: logs" in result.output

    # No clusters
    get_clusters.return_value = ClusterList([])
    result = runner.invoke(cluster_logs, ["NonExistentCluster"])

    assert result.exit_code == 1
    assert "Error: You don't have any clusters" in result.output
