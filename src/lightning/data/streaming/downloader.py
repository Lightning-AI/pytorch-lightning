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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from urllib import parse


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

    @abstractmethod
    def download_file(self, remote_chunkpath: str, local_chunkpath: str) -> None:
        pass


class S3Downloader(Downloader):
    @classmethod
    def download_file(cls, remote_filepath: str, local_filepath: str) -> None:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        extra_args: Dict[str, Any] = {}

        # Create a new session per thread
        session = boto3.session.Session()
        # Create a resource client using a thread's session object
        s3 = session.client("s3", config=Config(read_timeout=None))
        # Threads calling S3 operations return RuntimeError (cannot schedule new futures after
        # interpreter shutdown). Temporary solution is to have `use_threads` as `False`.
        # Issue: https://github.com/boto/boto3/issues/3113
        s3.download_file(
            obj.netloc,
            obj.path.lstrip("/"),
            local_filepath,
            ExtraArgs=extra_args,
            Config=TransferConfig(use_threads=False),
        )


class LocalDownloader(Downloader):
    @classmethod
    def download_file(cls, remote_filepath: str, local_filepath: str) -> None:
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError("The provided remote_path doesn't exist: {remote_path}")
        shutil.copy(remote_filepath, local_filepath)


_DOWNLOADERS = {"s3://": S3Downloader, "": LocalDownloader}


def get_downloader_cls(remote_dir: str) -> Type[Downloader]:
    for k, cls in _DOWNLOADERS.items():
        if remote_dir.startswith(k):
            return cls
    raise ValueError(f"The provided `remote_dir` {remote_dir} doesn't have a downloader associated.")
