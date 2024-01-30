import os
import sys
from pathlib import PosixPath
from unittest.mock import MagicMock

import pytest
from lightning.app.cli.commands import cp
from lightning.app.cli.commands.cd import _CD_FILE, cd
from lightning_cloud.openapi import (
    Externalv1Cluster,
    Externalv1LightningappInstance,
    V1CloudSpace,
    V1ClusterDriver,
    V1ClusterSpec,
    V1KubernetesClusterDriver,
    V1LightningappInstanceArtifact,
    V1LightningappInstanceSpec,
    V1ListCloudSpacesResponse,
    V1ListClustersResponse,
    V1ListLightningappInstanceArtifactsResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1ListProjectClusterBindingsResponse,
    V1Membership,
    V1ProjectClusterBinding,
    V1UploadProjectArtifactResponse,
)


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_local_to_remote(tmpdir, monkeypatch):
    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="project-0")]
    )

    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[Externalv1LightningappInstance(name="app-name-0", id="app-id-0")]
    )

    client.projects_service_list_project_cluster_bindings.return_value = V1ListProjectClusterBindingsResponse(
        clusters=[V1ProjectClusterBinding(cluster_id="my-cluster", cluster_name="my-cluster")]
    )

    result = MagicMock()
    result.get.return_value = V1UploadProjectArtifactResponse(urls=["http://foo.bar"])
    client.lightningapp_instance_service_upload_project_artifact.return_value = result

    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    assert cd("/", verify=False) == "/"
    cp.cp(str(tmpdir), "r:.")
    assert error_and_exit._mock_call_args_list[0].args[0] == "Uploading files at the project level isn't allowed yet."

    assert cd("/project-0/app-name-0", verify=False) == "/project-0/app-name-0"
    with open(f"{tmpdir}/a.txt", "w") as f:
        f.write("hello world !")

    file_uploader = MagicMock()
    monkeypatch.setattr(cp, "FileUploader", file_uploader)

    cp.cp(str(tmpdir), "r:.")
    assert file_uploader._mock_call_args[1]["name"] == f"{tmpdir}/a.txt"

    os.remove(_CD_FILE)


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_cloud_to_local(tmpdir, monkeypatch):
    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="project-0")]
    )

    clusters = MagicMock()
    clusters.clusters = [MagicMock()]
    client.projects_service_list_project_cluster_bindings.return_value = clusters

    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[Externalv1LightningappInstance(name="app-name-0", id="app-id-0")]
    )

    artifacts = [
        V1LightningappInstanceArtifact(filename=".file_1.txt", url="http://foo.bar/file_1.txt", size_bytes=123),
        V1LightningappInstanceArtifact(
            filename=".folder_1/file_2.txt", url="http://foo.bar/folder_1/file_2.txt", size_bytes=123
        ),
        V1LightningappInstanceArtifact(
            filename=".folder_2/folder_3/file_3.txt", url="http://foo.bar/folder_2/folder_3/file_3.txt", size_bytes=123
        ),
        V1LightningappInstanceArtifact(
            filename=".folder_4/file_4.txt", url="http://foo.bar/folder_4/file_4.txt", size_bytes=123
        ),
    ]

    client.lightningapp_instance_service_list_project_artifacts.return_value = (
        V1ListLightningappInstanceArtifactsResponse(artifacts=artifacts)
    )

    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    assert cd("/", verify=False) == "/"
    cp.cp(str(tmpdir), "r:.")
    assert error_and_exit._mock_call_args_list[0].args[0] == "Uploading files at the project level isn't allowed yet."

    assert cd("/project-0/app-name-0", verify=False) == "/project-0/app-name-0"

    download_file = MagicMock()
    monkeypatch.setattr(cp, "_download_file", download_file)

    cp.cp("r:.", str(tmpdir))

    assert len(download_file.call_args_list) == 4
    for i, call in enumerate(download_file.call_args_list):
        assert call.args[0] == PosixPath(tmpdir / artifacts[i].filename)
        assert call.args[1] == artifacts[i].url

    # cleanup
    os.remove(_CD_FILE)


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
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
    assert cd("/", verify=False) == "/"

    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=MagicMock()))
    cp.cp("./my-resource", "r:./my-resource", zip=True)
    error_and_exit.assert_called_once()
    assert "Zipping uploads isn't supported yet" in error_and_exit.call_args_list[0].args[0]


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_zip_src_path_too_short(monkeypatch):
    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=MagicMock()))
    cp.cp("r:/my-project", ".", zip=True)
    error_and_exit.assert_called_once()
    assert "The source path must be at least two levels deep" in error_and_exit.call_args_list[0].args[0]


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_cp_zip_remote_to_local_cloudspace_artifact(monkeypatch):
    assert cd("/", verify=False) == "/"

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
                    driver=V1ClusterDriver(kubernetes=V1KubernetesClusterDriver(root_domain_name="my-domain"))
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
    assert cd("/", verify=False) == "/"

    token_getter = MagicMock()
    token_getter._get_api_token.return_value = "my-token"
    monkeypatch.setattr(cp, "_AuthTokenGetter", MagicMock(return_value=token_getter))

    client = MagicMock()
    client.cluster_service_get_cluster.return_value = Externalv1Cluster(
        spec=V1ClusterSpec(driver=V1ClusterDriver(kubernetes=V1KubernetesClusterDriver(root_domain_name="my-domain")))
    )
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(name="my-project", project_id="my-project-id")]
    )
    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[
            Externalv1LightningappInstance(
                name="my-app", id="my-app-id", spec=V1LightningappInstanceSpec(cluster_id="my-cluster")
            )
        ]
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
