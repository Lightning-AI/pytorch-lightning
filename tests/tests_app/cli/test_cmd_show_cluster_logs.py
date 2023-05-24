from unittest import mock
from unittest.mock import MagicMock

from click.testing import CliRunner
from lightning_cloud.openapi import Externalv1Cluster

from lightning.app.cli.cmd_clusters import ClusterList
from lightning.app.cli.lightning_cli import cluster_logs


@mock.patch("lightning.app.cli.lightning_cli.LightningClient", MagicMock())
@mock.patch("lightning.app.cli.cmd_clusters.LightningClient", MagicMock())
@mock.patch("lightning.app.cli.lightning_cli.AWSClusterManager.get_clusters")
def test_show_logs_errors(get_clusters):
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

    # One cluster
    clusters = ClusterList([Externalv1Cluster(name="MyFakeCluster", id="MyFakeCluster")])
    get_clusters.return_value = clusters

    result = runner.invoke(cluster_logs, ["MyFakeClusterTwo"])

    assert result.exit_code == 1
    assert "Please select one of the following: [MyFakeCluster]" in str(result.output)
