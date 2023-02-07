from unittest.mock import MagicMock

from lightning_cloud.openapi import ProjectIdDataConnectionsBody, V1ListMembershipsResponse, V1Membership

from lightning.app.cli.connect import data


def test_connect_data(monkeypatch):

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="project-0", project_id="project-id-0"),
            V1Membership(name="project-1", project_id="project-id-1"),
            V1Membership(name="project 2", project_id="project-id-2"),
        ]
    )
    monkeypatch.setattr(data, "LightningClient", MagicMock(return_value=client))

    data.connect_data("imagenet", "s3://imagenet", destination="", project_name="project-0")

    client.data_connection_service_create_data_connection.assert_called_with(
        project_id="project-id-0",
        body=ProjectIdDataConnectionsBody(
            destination="",
            name="imagenet",
            source=":s3:/imagenet",
        ),
    )
