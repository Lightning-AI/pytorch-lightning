import platform
import sys
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning_cloud.openapi import (
    Externalv1Cluster,
    ProjectIdProjectclustersbindingsBody,
    V1BYOMClusterDriver,
    V1ClusterDriver,
    V1ClusterSpec,
    V1ClusterType,
    V1CreateClusterRequest,
    V1ListMembershipsResponse,
    V1Membership,
)

from lightning.app.cli.connect.maverick import deregister_from_cloud, register_to_cloud


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="lightning connect maverick is only supported on m1 mac",
)
def test_register_to_cloud(monkeypatch):
    mocked_client = MagicMock()
    monkeypatch.setattr("lightning.app.cli.connect.maverick.LightningClient", MagicMock(return_value=mocked_client))
    with pytest.raises(ValueError, match="Project project-0 does not exist."):
        register_to_cloud("maverick-001", "project-0")

    mocked_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="project-0", project_id="project-id-0"),
        ]
    )

    mocked_client.projects_service_list_project_cluster_bindings.return_value = MagicMock(clusters=[])
    register_to_cloud("maverick-001", "project-0")
    mocked_client.cluster_service_create_cluster.assert_called_with(
        body=V1CreateClusterRequest(
            name="maverick-001",
            spec=V1ClusterSpec(cluster_type=V1ClusterType.BYOM, driver=V1ClusterDriver(byom=V1BYOMClusterDriver())),
        )
    )

    mocked_client.projects_service_create_project_cluster_binding.assert_called_with(
        project_id="project-id-0",
        body=ProjectIdProjectclustersbindingsBody(cluster_id=mock.ANY),
    )


def test_register_to_cloud_without_project(monkeypatch):
    mocked_client = MagicMock()
    monkeypatch.setattr("lightning.app.cli.connect.maverick.LightningClient", MagicMock(return_value=mocked_client))
    mocked_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(project_id="project-id-0")]
    )
    mocked_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="project-0", project_id="project-id-0"),
        ]
    )

    mocked_client.projects_service_list_project_cluster_bindings.return_value = MagicMock(clusters=[])

    # calling without project name
    register_to_cloud("maverick-001", "")

    mocked_client.cluster_service_create_cluster.assert_called_with(
        body=V1CreateClusterRequest(
            name="maverick-001",
            spec=V1ClusterSpec(cluster_type=V1ClusterType.BYOM, driver=V1ClusterDriver(byom=V1BYOMClusterDriver())),
        )
    )

    mocked_client.projects_service_create_project_cluster_binding.assert_called_with(
        project_id="project-id-0",
        body=ProjectIdProjectclustersbindingsBody(cluster_id=mock.ANY),
    )


def test_deregister_from_cloud(monkeypatch):
    mocked_client = MagicMock()
    monkeypatch.setattr("lightning.app.cli.connect.maverick.LightningClient", MagicMock(return_value=mocked_client))
    mocked_client.cluster_service_list_clusters.return_value = MagicMock(
        clusters=[Externalv1Cluster(id="cluster-id-0", name="maverick-001")]
    )
    deregister_from_cloud("maverick-001")
    mocked_client.cluster_service_delete_cluster.assert_called_with(id="cluster-id-0", force=True)
