import sys
from unittest.mock import MagicMock

import pytest
from lightning_cloud.openapi import (
    Externalv1Cluster,
    Externalv1LightningappInstance,
    V1CloudSpace,
    V1ClusterDriver,
    V1ClusterSpec,
    V1GetClusterResponse,
    V1KubernetesClusterDriver,
    V1LightningappInstanceSpec,
    V1ListCloudSpacesResponse,
    V1ListClustersResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)

from lightning.app.cli.commands import cp, zip
from lightning.app.cli.commands.cd import cd


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_zip_local_to_remote(monkeypatch):
    assert "/" == cd("/", verify=False)

    error_and_exit = MagicMock()
    monkeypatch.setattr(zip, "_error_and_exit", error_and_exit)
    zip.zip(".", "r:.")
    error_and_exit.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_zip_remote_to_local_cloudspace_artifact(monkeypatch):
    assert "/" == cd("/", verify=False)

    token_getter = MagicMock()
    token_getter._get_api_token.return_value = "my-token"
    monkeypatch.setattr(zip, "_AuthTokenGetter", MagicMock(return_value=token_getter))

    client = MagicMock()
    client.cluster_service_list_clusters.return_value = V1ListClustersResponse(
        default_cluster="my-cluster",
        clusters=[
            Externalv1Cluster(
                id="my-cluster",
                spec=V1ClusterSpec(
                    driver=V1ClusterDriver(
                        kubernetes=V1KubernetesClusterDriver(
                            root_domain_name="my-domain",
                        ),
                    ),
                ),
            )
        ],
    )
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="my-project", project_id="my-project-id")]
    )
    client.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="my-cloudspace", id="my-cloudspace-id")],
    )
    monkeypatch.setattr(zip, "LightningClient", MagicMock(return_value=client))
    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    download_file = MagicMock()
    monkeypatch.setattr(zip, "_download_file", download_file)

    cloudspace_artifact = "r:/my-project/my-cloudspace/my-artifact"
    zip.zip(cloudspace_artifact, ".")

    download_file.assert_called_once()
    assert download_file.call_args_list[0].args[0] == "./my-artifact.zip"
    assert (
        download_file.call_args_list[0].args[1]
        == "https://storage.my-domain/v1/projects/my-project-id/artifacts/download?prefix=/cloudspaces/my-cloudspace-id/my-artifact&token=my-token"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_zip_remote_to_local_app_artifact(monkeypatch):
    assert "/" == cd("/", verify=False)

    token_getter = MagicMock()
    token_getter._get_api_token.return_value = "my-token"
    monkeypatch.setattr(zip, "_AuthTokenGetter", MagicMock(return_value=token_getter))

    client = MagicMock()
    client.cluster_service_get_cluster.return_value = V1GetClusterResponse(
        spec=V1ClusterSpec(
            driver=V1ClusterDriver(
                kubernetes=V1KubernetesClusterDriver(
                    root_domain_name="my-domain",
                ),
            ),
        ),
    )
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="my-project", project_id="my-project-id")]
    )
    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[
            Externalv1LightningappInstance(
                name="my-app",
                id="my-app-id",
                spec=V1LightningappInstanceSpec(
                    cluster_id="my-cluster",
                ),
            )
        ],
    )
    monkeypatch.setattr(zip, "LightningClient", MagicMock(return_value=client))
    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    download_file = MagicMock()
    monkeypatch.setattr(zip, "_download_file", download_file)

    app_artifact = "r:/my-project/my-app/my-artifact"
    zip.zip(app_artifact, ".")

    download_file.assert_called_once()
    assert download_file.call_args_list[0].args[0] == "./my-artifact.zip"
    assert (
        download_file.call_args_list[0].args[1]
        == "https://storage.my-domain/v1/projects/my-project-id/artifacts/download?prefix=/lightningapps/my-app-id/my-artifact&token=my-token"
    )
