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

import time

import requests
from requests.adapters import HTTPAdapter
from rich.progress import BarColumn, Progress, TextColumn
from urllib3.util.retry import Retry


class FileUploader:
    """This class uploads a source file with presigned url to S3.

    Attributes
    ----------
    source_file: str
        Source file to upload
    presigned_url: str
        Presigned urls dictionary, with key as part number and values as urls
    retries: int
        Amount of retries when requests encounter an error
    total_size: int
        Size of all files to upload
    name: str
        Name of this upload to display progress
    """

    workers: int = 8
    retries: int = 10000
    disconnect_retry_wait_seconds: int = 5
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[self.progress.percentage]{task.percentage:>3.1f}%",
    )

    def __init__(self, presigned_url: str, source_file: str, total_size: int, name: str, use_progress: bool = True):
        self.presigned_url = presigned_url
        self.source_file = source_file
        self.total_size = total_size
        self.name = name
        self.use_progress = use_progress
        self.task_id = None

    def upload_data(self, url: str, data: bytes, retries: int, disconnect_retry_wait_seconds: int) -> str:
        """Send data to url.

        Parameters
        ----------
        url: str
            url string to send data to
        data: bytes
             Bytes of data to send to url
        retries: int
            Amount of retries
        disconnect_retry_wait_seconds: int
            Amount of seconds between disconnect retry

        Returns
        -------
        str
            ETag from response
        """
        disconnect_retries = retries
        while disconnect_retries > 0:
            try:
                retries = Retry(total=10)
                with requests.Session() as s:
                    s.mount("https://", HTTPAdapter(max_retries=retries))
                    return self._upload_data(s, url, data)
            except BrokenPipeError:
                time.sleep(disconnect_retry_wait_seconds)
                disconnect_retries -= 1

        raise ValueError("Unable to upload file after multiple attempts")

    def _upload_data(self, s: requests.Session, url: str, data: bytes):
        resp = s.put(url, data=data)
        if "ETag" not in resp.headers:
            raise ValueError(f"Unexpected response from {url}, response: {resp.content}")
        return resp.headers["ETag"]

    def upload(self) -> None:
        """Upload files from source dir into target path in S3."""
        no_task = self.task_id is None
        if self.use_progress and no_task:
            self.task_id = self.progress.add_task("upload", filename=self.name, total=self.total_size)
            self.progress.start()
        try:
            with open(self.source_file, "rb") as f:
                data = f.read()
            self.upload_data(self.presigned_url, data, self.retries, self.disconnect_retry_wait_seconds)
            if self.use_progress:
                self.progress.update(self.task_id, advance=len(data))
        finally:
            if self.use_progress and no_task:
                self.progress.stop()
