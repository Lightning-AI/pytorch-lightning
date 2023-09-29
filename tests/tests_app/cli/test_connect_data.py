import sys
from unittest.mock import MagicMock

import pytest
from lightning.app.cli.connect import data


@pytest.mark.skipif(sys.platform == "win32", reason="lightning connect data isn't supported on windows")
def test_connect_data_no_project(monkeypatch):
    from lightning_cloud.openapi import V1ListMembershipsResponse, V1Membership

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(memberships=[])
    monkeypatch.setattr(data, "LightningClient", MagicMock(return_value=client))

    _error_and_exit = MagicMock()
    monkeypatch.setattr(data, "_error_and_exit", _error_and_exit)

    _get_project = MagicMock()
    _get_project.return_value = V1Membership(name="project-0", project_id="project-id-0")
    monkeypatch.setattr(data, "_get_project", _get_project)

    data.connect_data("imagenet", region="us-east-1", source="imagenet", destination="", project_name="project-0")

    _get_project.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="lightning connect data isn't supported on windows")
def test_connect_data(monkeypatch):
    from lightning_cloud.openapi import Create, V1AwsDataConnection, V1ListMembershipsResponse, V1Membership

    client = MagicMock()
    client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[
            V1Membership(name="project-0", project_id="project-id-0"),
            V1Membership(name="project-1", project_id="project-id-1"),
            V1Membership(name="project 2", project_id="project-id-2"),
        ]
    )
    monkeypatch.setattr(data, "LightningClient", MagicMock(return_value=client))

    _error_and_exit = MagicMock()
    monkeypatch.setattr(data, "_error_and_exit", _error_and_exit)
    data.connect_data("imagenet", region="us-east-1", source="imagenet", destination="", project_name="project-0")

    _error_and_exit.assert_called_with(
        "Only public S3 folders are supported for now. Please, open a Github issue with your use case."
    )

    data.connect_data("imagenet", region="us-east-1", source="s3://imagenet", destination="", project_name="project-0")

    client.data_connection_service_create_data_connection.assert_called_with(
        project_id="project-id-0",
        body=Create(
            name="imagenet",
            aws=V1AwsDataConnection(destination="", region="us-east-1", source="s3://imagenet", secret_arn_name=""),
        ),
    )
