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

import requests
from lightning_cloud.openapi import AuthServiceApi, ModelsStoreApi, ProjectsServiceApi
from lightning_cloud.rest_client import create_swagger_client
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper


def _upload_file_to_url(url: str, path: str, progress_bar: bool) -> None:
    if progress_bar:
        file_size = os.path.getsize(path)
        with open(path, "rb") as fd, tqdm(
            desc="Uploading",
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1000,
        ) as t:
            reader_wrapper = CallbackIOWrapper(t.update, fd, "read")
            response = requests.put(url, data=reader_wrapper)
            response.raise_for_status()
    else:
        with open(path, "rb") as fo:
            requests.put(url, data=fo)


def _download_file_from_url(url: str, path: str, progress_bar: bool) -> None:
    with requests.get(url, stream=True) as req_stream:
        total_size_in_bytes = int(req_stream.headers.get("content-length", 0))
        block_size = 1000 * 1000  # 1 MB

        download_progress_bar = None
        if progress_bar:
            download_progress_bar = tqdm(
                desc="Downloading",
                total=total_size_in_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1000,
            )
        with open(path, "wb") as f:
            for chunk in req_stream.iter_content(chunk_size=block_size):
                if download_progress_bar:
                    download_progress_bar.update(len(chunk))
                f.write(chunk)
        if download_progress_bar:
            download_progress_bar.close()


class _Client(AuthServiceApi, ModelsStoreApi, ProjectsServiceApi):
    def __init__(self):
        api_client = create_swagger_client()
        super().__init__(api_client)
