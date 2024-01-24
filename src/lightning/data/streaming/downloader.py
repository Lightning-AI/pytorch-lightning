# Copyright The Lightning AI team.
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
import shutil
import subprocess
from abc import ABC
from typing import Any, Dict, List
from urllib import parse

from filelock import FileLock, Timeout

from lightning.data.streaming.client import S3Client


class Downloader(ABC):
    def __init__(self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]]):
        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks

    def download_chunk_from_index(self, chunk_index: int) -> None:
        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)
        self.download_file(remote_chunkpath, local_chunkpath)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str) -> None:
        pass


class S3Downloader(Downloader):
    def __init__(self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]]):
        super().__init__(remote_dir, cache_dir, chunks)
        self._s5cmd_available = os.system("s5cmd > /dev/null 2>&1") == 0

        if not self._s5cmd_available:
            self._client = S3Client()

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        try:
            with FileLock(local_filepath + ".lock", timeout=0):
                if self._s5cmd_available:
                    proc = subprocess.Popen(
                        f"s5cmd cp {remote_filepath} {local_filepath}",
                        shell=True,
                        stdout=subprocess.PIPE,
                    )
                    proc.wait()
                else:
                    from boto3.s3.transfer import TransferConfig

                    extra_args: Dict[str, Any] = {}

                    # try:
                    #     with FileLock(local_filepath + ".lock", timeout=1):
                    if not os.path.exists(local_filepath):
                        # Issue: https://github.com/boto/boto3/issues/3113
                        self._client.client.download_file(
                            obj.netloc,
                            obj.path.lstrip("/"),
                            local_filepath,
                            ExtraArgs=extra_args,
                            Config=TransferConfig(use_threads=False),
                        )
        except Timeout:
            # another process is responsible to download that file, continue
            pass


class LocalDownloader(Downloader):
    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError(f"The provided remote_path doesn't exist: {remote_filepath}")

        if remote_filepath != local_filepath and not os.path.exists(local_filepath):
            shutil.copy(remote_filepath, local_filepath)


_DOWNLOADERS = {"s3://": S3Downloader, "": LocalDownloader}


def get_downloader_cls(remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]]) -> Downloader:
    for k, cls in _DOWNLOADERS.items():
        if str(remote_dir).startswith(k):
            return cls(remote_dir, cache_dir, chunks)
    raise ValueError(f"The provided `remote_dir` {remote_dir} doesn't have a downloader associated.")
