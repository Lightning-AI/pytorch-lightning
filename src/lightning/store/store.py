# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List

from lightning_cloud.openapi import V1Model, V1UploadModelRequest

from lightning.app.utilities.cloud import _get_project
from lightning.store.utils import _Client, _download_file_from_url, _upload_file_to_url


def upload_model(
    name: str,
    path: str,
    version: str = "latest",
    progress_bar: bool = True,
) -> None:
    """Upload a model to the lightning cloud.

    Args:
        name:
            The model name.
        path:
            The path to the checkpoint to be uploaded.
        version:
            The version of the model to be uploaded. If not provided, default will be latest (not overridden).
        progress_bar:
            A progress bar to show the uploading status. Disable this if not needed, by setting to `False`.

    """
    client = _Client()
    user = client.auth_service_get_user()
    # TODO: Allow passing this
    project_id = _get_project(client).project_id

    # TODO: Post model parts if the file size is over threshold
    body = V1UploadModelRequest(
        name=f"{user.username}/{name}",
        version=version,
        project_id=project_id,
    )
    model = client.models_store_upload_model(body)

    _upload_file_to_url(model.upload_url, path, progress_bar=progress_bar)


def download_model(
    name: str,
    path: str,
    version: str = "latest",
    progress_bar: bool = True,
) -> None:
    """Download a model from the lightning cloud.

    Args:
        name:
            The unique name of the model to be downloaded. Format: `<username>/<model_name>`.
        path:
            The path to download the model to.
        version:
            The version of the model to be uploaded. If not provided, default will be latest (not overridden).
        progress_bar:
            Show progress on download.

    """
    client = _Client()
    download_url = client.models_store_download_model(name=name, version=version).download_url
    _download_file_from_url(download_url, os.path.abspath(path), progress_bar=progress_bar)


def list_models() -> List[V1Model]:
    """List your models in the lightning cloud.

    Returns:
        A list of model objects.

    """
    client = _Client()
    # TODO: Allow passing this
    project_id = _get_project(client).project_id
    return client.models_store_list_models(project_id=project_id).models
