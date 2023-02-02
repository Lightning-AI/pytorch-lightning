import os
from unittest.mock import MagicMock

from lightning_cloud.openapi import (
    Externalv1LightningappInstance,
    V1LightningappInstanceArtifact,
    V1ListLightningappInstanceArtifactsResponse,
    V1ListLightningappInstancesResponse,
    V1ListMembershipsResponse,
    V1Membership,
)

from lightning.app.cli.commands import ls
from lightning.app.cli.commands.cd import _CD_FILE, cd


def test_ls(monkeypatch):
    """This test validates ls behaves as expected."""
    assert "/" == cd("/")

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="project-0", project_id="project-id-0"),
            V1Membership(name="project-1", project_id="project-id-1"),
        ]
    )

    client.lightningapp_instance_service_list_lightningapp_instances.return_value = V1ListLightningappInstancesResponse(
        lightningapps=[
            Externalv1LightningappInstance(
                name="app-name-0",
                id="app-id-0",
            ),
            Externalv1LightningappInstance(
                name="app-name-1",
                id="app-id-1",
            ),
        ]
    )

    client.lightningapp_instance_service_list_lightningapp_instance_artifacts.return_value = (
        V1ListLightningappInstanceArtifactsResponse(
            artifacts=[
                V1LightningappInstanceArtifact(
                    filename="file_1.txt",
                ),
                V1LightningappInstanceArtifact(
                    filename="folder_1/file_2.txt",
                ),
                V1LightningappInstanceArtifact(
                    filename="folder_2/folder_3/file_3.txt",
                ),
                V1LightningappInstanceArtifact(
                    filename="folder_2/file_4.txt",
                ),
            ]
        )
    )

    monkeypatch.setattr(ls, "LightningClient", MagicMock(return_value=client))

    assert ls.ls() == ["[blue]project-0[/blue]", "[blue]project-1[/blue]"]
    assert "/project-0" == cd("project-0")

    assert ls.ls() == ["[blue]app-name-0[/blue]", "[blue]app-name-1[/blue]"]
    assert "/project-0/app-name-1" == cd("app-name-1")
    assert ls.ls() == ["[blue]folder_1[/blue]", "[blue]folder_2[/blue]", "[white]file_1.txt[/white]"]
    assert "/project-0/app-name-1/folder_1" == cd("folder_1")
    assert ls.ls() == ["[white]file_2.txt[/white]"]
    assert "/project-0/app-name-1/folder_2" == cd("../folder_2")
    assert ls.ls() == ["[blue]folder_3[/blue]", "[white]file_4.txt[/white]"]
    assert "/project-0/app-name-1/folder_2/folder_3" == cd("folder_3")
    assert ls.ls() == ["[white]file_3.txt[/white]"]

    os.remove(_CD_FILE)
