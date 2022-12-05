from unittest import mock

from lightning_cloud.openapi import Externalv1Cluster, V1ClusterSpec, V1ClusterType

from lightning_app.cli.lightning_cli_delete import _find_cluster_for_user


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
def test_cli_delete_app_find_cluster_without_valid_cluster_id_asks_if_they_meant_to_use_valid(
    list_clusters: mock.MagicMock,
):
    pass


def test_cli_delete_app_find_selected_app_instance_id_exists():
    pass


def test_cli_delete_app_find_selected_app_instance_id_does_not_exist():
    pass


def test_appmanager_delete_calls_lightningapp_instance_service_delete_lightningapp_instance_with_correct_args():
    pass
