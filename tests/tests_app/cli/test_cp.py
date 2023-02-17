import os
import shutil
import sys
from unittest.mock import MagicMock

import pytest
import requests
from lightning_cloud.openapi import (
    Externalv1Cluster,
    Externalv1LightningappInstance,
    V1CloudSpace,
    V1ClusterDriver,
    V1ClusterSpec,
    V1GetClusterResponse,
    V1KubernetesClusterDriver,
    V1LightningappInstanceArtifact,
    V1LightningappInstanceSpec,
    V1ListCloudSpacesResponse,
    V1ListClustersResponse,
    V1ListLightningappInstanceArtifactsResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)

from lightning.app.cli.commands import cp
from lightning.app.cli.commands.cd import _CD_FILE, cd


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_local_to_remote(tmpdir, monkeypatch):
    assert "/" == cd("/", verify=False)

    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    cp.cp(str(tmpdir), "r:.")
    assert error_and_exit._mock_call_args_list[0].args[0] == "Uploading files at the project level isn't allowed yet."

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="project-0")]
    )

    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[
            Externalv1LightningappInstance(
                name="app-name-0",
                id="app-id-0",
            )
        ]
    )

    clusters = MagicMock()
    clusters.clusters = [MagicMock()]
    client.projects_service_list_project_cluster_bindings.return_value = clusters

    client.lightningapp_instance_service_upload_project_artifact.return_value = MagicMock()

    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    assert "/project-0/app-name-0" == cd("/project-0/app-name-0", verify=False)

    with open(f"{tmpdir}/a.txt", "w") as f:
        f.write("hello world !")

    file_uploader = MagicMock()
    monkeypatch.setattr(cp, "FileUploader", file_uploader)

    cp.cp(str(tmpdir), "r:.")

    assert file_uploader._mock_call_args[1]["name"] == "" f"{tmpdir}/a.txt"

    os.remove(_CD_FILE)


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_cloud_to_local(tmpdir, monkeypatch):
    assert "/" == cd("/", verify=False)

    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    cp.cp(str(tmpdir), "r:.")
    assert error_and_exit._mock_call_args_list[0].args[0] == "Uploading files at the project level isn't allowed yet."

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="project-0")]
    )

    clusters = MagicMock()
    clusters.clusters = [MagicMock()]
    client.projects_service_list_project_cluster_bindings.return_value = clusters

    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[
            Externalv1LightningappInstance(
                name="app-name-0",
                id="app-id-0",
            )
        ]
    )

    client.lightningapp_instance_service_list_project_artifacts.return_value = (
        V1ListLightningappInstanceArtifactsResponse(
            artifacts=[
                V1LightningappInstanceArtifact(
                    filename=".file_1.txt",
                    size_bytes=123,
                ),
                V1LightningappInstanceArtifact(
                    filename=".folder_1/file_2.txt",
                    size_bytes=123,
                ),
                V1LightningappInstanceArtifact(
                    filename=".folder_2/folder_3/file_3.txt",
                    size_bytes=123,
                ),
                V1LightningappInstanceArtifact(
                    filename=".folder_4/file_4.txt",
                    size_bytes=123,
                ),
            ]
        )
    )

    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    assert "/project-0/app-name-0" == cd("/project-0/app-name-0", verify=False)

    get_fn = requests.get

    def patch_get(*args, **kwargs):
        return get_fn("https://pl-flash-data.s3.amazonaws.com/daef0454-97a4-4a22-a704-fb9f80b7ea83.txt")

    monkeypatch.setattr(requests, "get", patch_get)

    cp.cp("r:.", str(tmpdir))
    cp.cp("r:.", ".")
    cp.cp("r:.", "test_cp_cloud_to_local")

    # cleanup
    os.remove(".file_1.txt")
    shutil.rmtree(".folder_1")
    shutil.rmtree(".folder_2")
    shutil.rmtree(".folder_4")
    shutil.rmtree("test_cp_cloud_to_local")
    os.remove(_CD_FILE)


def test_sanitize_path():
    path, is_remote = cp._sanitize_path("r:default-project", "/")
    assert path == "/default-project"
    assert is_remote

    path, _ = cp._sanitize_path("r:foo", "/default-project")
    assert path == "/default-project/foo"

    path, _ = cp._sanitize_path("foo", "/default-project")
    assert path == "foo"


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_zip_arg_order(monkeypatch):
    assert "/" == cd("/", verify=False)

    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    cp.cp("./my-resource", "r:./my-resource", zip=True)
    error_and_exit.assert_called_once()
    assert "Zipping uploads isn't supported yet" in error_and_exit.call_args_list[0].args[0]


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_zip_src_path_too_short(monkeypatch):
    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    cp.cp("r:/my-project", ".", zip=True)
    error_and_exit.assert_called_once()
    assert "The source path must be at least two levels deep" in error_and_exit.call_args_list[0].args[0]


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_zip_remote_to_local_cloudspace_artifact(monkeypatch):
    assert "/" == cd("/", verify=False)

    token_getter = MagicMock()
    token_getter._get_api_token.return_value = "my-token"
    monkeypatch.setattr(cp, "_AuthTokenGetter", MagicMock(return_value=token_getter))

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
    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    download_file = MagicMock()
    monkeypatch.setattr(cp, "_download_file", download_file)

    cloudspace_artifact = "r:/my-project/my-cloudspace/my-artifact"
    cp.cp(cloudspace_artifact, ".", zip=True)

    download_file.assert_called_once()
    assert download_file.call_args_list[0].args[0] == "./my-artifact.zip"
    assert (
        download_file.call_args_list[0].args[1]
        == "https://storage.my-domain/v1/projects/my-project-id/artifacts/download"
        + "?prefix=/cloudspaces/my-cloudspace-id/my-artifact&token=my-token"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_zip_remote_to_local_app_artifact(monkeypatch):
    assert "/" == cd("/", verify=False)

    token_getter = MagicMock()
    token_getter._get_api_token.return_value = "my-token"
    monkeypatch.setattr(cp, "_AuthTokenGetter", MagicMock(return_value=token_getter))

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
    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    download_file = MagicMock()
    monkeypatch.setattr(cp, "_download_file", download_file)

    app_artifact = "r:/my-project/my-app/my-artifact"
    cp.cp(app_artifact, ".", zip=True)

    download_file.assert_called_once()
    assert download_file.call_args_list[0].args[0] == "./my-artifact.zip"
    assert (
        download_file.call_args_list[0].args[1]
        == "https://storage.my-domain/v1/projects/my-project-id/artifacts/download"
        + "?prefix=/lightningapps/my-app-id/my-artifact&token=my-token"
    )
