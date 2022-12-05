from unittest import mock

from lightning_cloud.openapi import Externalv1Cluster, Externalv1LightningappInstance, V1ClusterSpec, V1ClusterType

from lightning_app.cli.lightning_cli_delete import _find_cluster_for_user, _find_selected_app_instance_id


@mock.patch("lightning_app.cli.lightning_cli_delete.AWSClusterManager.list_clusters")
def test_find_cluster_for_user_when_provided_valid_cluster_id(list_clusters_mock: mock.MagicMock):
    list_clusters_mock.return_value = [
        Externalv1Cluster(
            id="default",
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.GLOBAL,
            ),
        ),
        Externalv1Cluster(
            id="custom",
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.BYOC,
            ),
        ),
    ]
    returned_cluster_id = _find_cluster_for_user(app_name="whatever", cluster_id="custom")
    assert returned_cluster_id == "custom"


@mock.patch("lightning_app.cli.lightning_cli_delete.AWSClusterManager.list_clusters")
def test_find_cluster_for_user_without_cluster_id_uses_default(list_clusters_mock: mock.MagicMock):
    list_clusters_mock.return_value = [
        Externalv1Cluster(
            id="default",
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.GLOBAL,
            ),
        )
    ]
    returned_cluster_id = _find_cluster_for_user(app_name="whatever", cluster_id=None)
    assert returned_cluster_id == "default"


@mock.patch("lightning_app.cli.lightning_cli_delete.AWSClusterManager.list_clusters")
@mock.patch("lightning_app.cli.lightning_cli_delete.inquirer")
def test_find_cluster_for_user_without_valid_cluster_id_asks_if_they_meant_to_use_valid(
    list_clusters_mock: mock.MagicMock,
    inquirer_mock: mock.MagicMock,
):
    list_clusters_mock.return_value = [
        Externalv1Cluster(
            id="default",
            spec=V1ClusterSpec(
                cluster_type=V1ClusterType.GLOBAL,
            ),
        )
    ]
    _find_cluster_for_user(app_name="whatever", cluster_id="does-not-exist")
    inquirer_mock.assert_called()


@mock.patch("lightning_app.cli.lightning_cli_delete._AppManager.list_apps")
def test_app_find_selected_app_instance_id_when_app_name_exists(list_apps_mock: mock.MagicMock):
    list_apps_mock.return_value = [
        Externalv1LightningappInstance(name="app-name", id="app-id"),
    ]
    returned_app_instance_id = _find_selected_app_instance_id(app_name="app-name", cluster_id="default-cluster")
    assert returned_app_instance_id == "app-id"
    list_apps_mock.assert_called_once_with(cluster_id="default-cluster")


@mock.patch("lightning_app.cli.lightning_cli_delete._AppManager.list_apps")
def test_app_find_selected_app_instance_id_when_app_id_exists(list_apps_mock: mock.MagicMock):
    list_apps_mock.return_value = [
        Externalv1LightningappInstance(name="app-name", id="app-id"),
    ]
    returned_app_instance_id = _find_selected_app_instance_id(app_name="app-id", cluster_id="default-cluster")
    assert returned_app_instance_id == "app-id"
    list_apps_mock.assert_called_once_with(cluster_id="default-cluster")
