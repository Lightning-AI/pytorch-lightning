import os
from unittest import mock

from lightning.store import download_model, list_models, upload_model
from lightning_cloud.openapi import (
    V1DownloadModelResponse,
    V1GetUserResponse,
    V1ListMembershipsResponse,
    V1ListModelsResponse,
    V1Membership,
    V1Model,
    V1Project,
    V1UploadModelRequest,
    V1UploadModelResponse,
)


@mock.patch("lightning.store.store._Client")
@mock.patch("lightning.store.store._upload_file_to_url")
def test_upload_model(mock_upload_file_to_url, mock_client):
    mock_client = mock_client()

    mock_client.auth_service_get_user.return_value = V1GetUserResponse(username="test-username")

    # either one of these project APIs could be called
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(project_id="test-project-id")],
    )
    mock_client.projects_service_get_project.return_value = V1Project(id="test-project-id")

    mock_client.models_store_upload_model.return_value = V1UploadModelResponse(
        upload_url="https://test",
    )

    upload_model("test-model", "test.ckpt", version="0.0.1")

    mock_client.auth_service_get_user.assert_called_once()
    mock_client.models_store_upload_model.assert_called_once_with(
        V1UploadModelRequest(
            name="test-username/test-model",
            version="0.0.1",
            project_id="test-project-id",
        )
    )

    mock_upload_file_to_url.assert_called_once_with("https://test", "test.ckpt", progress_bar=True)


@mock.patch("lightning.store.store._Client")
@mock.patch("lightning.store.store._download_file_from_url")
def test_download_model(mock_download_file_from_url, mock_client):
    mock_client = mock_client()

    mock_client.models_store_download_model.return_value = V1DownloadModelResponse(
        download_url="https://test",
    )

    download_model("test-username/test-model", "test.ckpt", version="0.0.1")

    mock_client.models_store_download_model.assert_called_once_with(
        name="test-username/test-model",
        version="0.0.1",
    )

    mock_download_file_from_url.assert_called_once_with("https://test", os.path.abspath("test.ckpt"), progress_bar=True)


@mock.patch("lightning.store.store._Client")
def test_list_models(mock_client):
    mock_client = mock_client()

    # either one of these project APIs could be called
    mock_client.projects_service_list_memberships.return_value = V1ListMembershipsResponse(
        memberships=[V1Membership(project_id="test-project-id")],
    )
    mock_client.projects_service_get_project.return_value = V1Project(id="test-project-id")

    mock_client.models_store_list_models.return_value = V1ListModelsResponse(models=[V1Model(name="test-model")])

    res = list_models()
    assert res[0].name == "test-model"

    mock_client.models_store_list_models.assert_called_once_with(project_id="test-project-id")
