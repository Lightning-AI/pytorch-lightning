import os
from unittest.mock import MagicMock

import requests
from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1LightningappInstanceArtifact,
    V1ListLightningappInstanceArtifactsResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)

from lightning.app.cli.commands import cp
from lightning.app.cli.commands.cd import _CD_FILE, cd


def test_cp_local_to_remote(tmpdir, monkeypatch):
    assert "/" == cd("/")

    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    cp.cp(tmpdir, "r:.")
    assert error_and_exit._mock_call_args_list[0].args[0] == "Uploading files at the project level isn't supported yet."

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

    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    assert "/project-0/app-name-0" == cd("/project-0/app-name-0")

    with open(f"{tmpdir}/a.txt", "w") as f:
        f.write("hello world !")

    file_uploader = MagicMock()
    monkeypatch.setattr(cp, "FileUploader", file_uploader)

    cp.cp(str(tmpdir), "r:.")

    assert file_uploader._mock_call_args[1]["name"] == "" f"{tmpdir}/a.txt"

    os.remove(_CD_FILE)


def test_cp_cloud_to_local(tmpdir, monkeypatch):
    assert "/" == cd("/")

    error_and_exit = MagicMock()
    monkeypatch.setattr(cp, "_error_and_exit", error_and_exit)
    cp.cp(tmpdir, "r:.")
    assert error_and_exit._mock_call_args_list[0].args[0] == "Uploading files at the project level isn't supported yet."

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

    client.lightningapp_instance_service_list_lightningapp_instance_artifacts.return_value = (
        V1ListLightningappInstanceArtifactsResponse(
            artifacts=[
                V1LightningappInstanceArtifact(
                    filename="file_1.txt",
                    size_bytes=123,
                ),
                V1LightningappInstanceArtifact(
                    filename="folder_1/file_2.txt",
                    size_bytes=123,
                ),
                V1LightningappInstanceArtifact(
                    filename="folder_2/folder_3/file_3.txt",
                    size_bytes=123,
                ),
                V1LightningappInstanceArtifact(
                    filename="folder_2/file_4.txt",
                    size_bytes=123,
                ),
            ]
        )
    )

    monkeypatch.setattr(cp, "LightningClient", MagicMock(return_value=client))

    assert "/project-0/app-name-0" == cd("/project-0/app-name-0")

    get_fn = requests.get

    def patch_get(*args, **kwargs):
        return get_fn("https://pl-flash-data.s3.amazonaws.com/daef0454-97a4-4a22-a704-fb9f80b7ea83.txt")

    monkeypatch.setattr(requests, "get", patch_get)

    cp.cp("r:.", str(tmpdir))

    os.remove(_CD_FILE)
