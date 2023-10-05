import os
import sys
from unittest.mock import MagicMock

import pytest
from lightning.app.cli.commands import cd, ls
from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1LightningappInstanceArtifact,
    V1ListCloudSpacesResponse,
    V1ListLightningappInstanceArtifactsResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows yet")
def test_ls(monkeypatch):
    """This test validates ls behaves as expected."""
    if os.path.exists(cd._CD_FILE):
        os.remove(cd._CD_FILE)

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="project-0", project_id="project-id-0"),
            V1Membership(name="project-1", project_id="project-id-1"),
            V1Membership(name="project 2", project_id="project-id-2"),
        ]
    )

    client.lightningapp_instance_service_list_lightningapp_instances().get.return_value = (
        V1ListLightningappInstancesResponse(
            lightningapps=[
                Externalv1LightningappInstance(name="app-name-0", id="app-id-0"),
                Externalv1LightningappInstance(name="app-name-1", id="app-id-1"),
                Externalv1LightningappInstance(name="app name 2", id="app-id-1"),
            ]
        )
    )

    client.cloud_space_service_list_cloud_spaces().get.return_value = V1ListCloudSpacesResponse(cloudspaces=[])

    clusters = MagicMock()
    clusters.clusters = [MagicMock()]
    client.projects_service_list_project_cluster_bindings.return_value = clusters

    def fn(*args, prefix, **kwargs):
        splits = [split for split in prefix.split("/") if split != ""]
        if len(splits) == 2:
            return V1ListLightningappInstanceArtifactsResponse(
                artifacts=[
                    V1LightningappInstanceArtifact(filename="file_1.txt"),
                    V1LightningappInstanceArtifact(filename="folder_1/file_2.txt"),
                    V1LightningappInstanceArtifact(filename="folder_2/folder_3/file_3.txt"),
                    V1LightningappInstanceArtifact(filename="folder_2/file_4.txt"),
                ]
            )
        if splits[-1] == "folder_1":
            return V1ListLightningappInstanceArtifactsResponse(
                artifacts=[V1LightningappInstanceArtifact(filename="file_2.txt")]
            )
        if splits[-1] == "folder_2":
            return V1ListLightningappInstanceArtifactsResponse(
                artifacts=[
                    V1LightningappInstanceArtifact(filename="folder_3/file_3.txt"),
                    V1LightningappInstanceArtifact(filename="file_4.txt"),
                ]
            )
        if splits[-1] == "folder_3":
            return V1ListLightningappInstanceArtifactsResponse(
                artifacts=[
                    V1LightningappInstanceArtifact(filename="file_3.txt"),
                ]
            )
        return None

    client.lightningapp_instance_service_list_project_artifacts = fn

    monkeypatch.setattr(ls, "LightningClient", MagicMock(return_value=client))

    assert ls.ls() == ["project-0", "project-1", "project 2"]
    assert cd.cd("project-0", verify=False) == "/project-0"

    assert ls.ls() == ["app name 2", "app-name-0", "app-name-1"]
    assert f"/project-0{os.sep}app-name-1" == cd.cd("app-name-1", verify=False)
    assert ls.ls() == ["file_1.txt", "folder_1", "folder_2"]
    assert f"/project-0{os.sep}app-name-1{os.sep}folder_1" == cd.cd("folder_1", verify=False)
    assert ls.ls() == ["file_2.txt"]
    assert f"/project-0{os.sep}app-name-1{os.sep}folder_2" == cd.cd("../folder_2", verify=False)
    assert ls.ls() == ["folder_3", "file_4.txt"]
    assert f"/project-0{os.sep}app-name-1{os.sep}folder_2{os.sep}folder_3" == cd.cd("folder_3", verify=False)
    assert ls.ls() == ["file_3.txt"]

    assert cd.cd("/project 2", verify=False) == "/project 2"
    assert ls.ls() == ["app name 2", "app-name-0", "app-name-1"]
    assert cd.cd("app name 2", verify=False) == "/project 2/app name 2"
    assert ls.ls() == ["file_1.txt", "folder_1", "folder_2"]

    os.remove(cd._CD_FILE)
