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

import json
import os
import subprocess
from subprocess import Popen
from typing import Optional, Tuple
from urllib import parse

from lightning.data.cache.pytree import treespec_loads
from lightning.data.cache.sampler import ChunkedIndex


class ChunksConfig:
    def __init__(self, cache_dir: str, index_filenames: str, _remote_dir: Optional[str]):
        self._cache_dir = cache_dir
        self.index_filenames = sorted(index_filenames)
        self._intervals = []
        self._config = None
        self._chunks = []
        self._remote_dir = _remote_dir

        for filename in self.index_filenames:
            with open(os.path.join(self._cache_dir, filename)) as f:
                data = json.load(f)

                if self._config is None:
                    self._config = data["config"]

                elif self._config != data["config"]:
                    raise Exception("The config isn't consistent between chunks. This shouldn't have happened.")

                self._chunks.extend(data["chunks"])

        self._config["data_spec"] = treespec_loads(self._config["data_spec"])

        for chunk in self._chunks:
            start, end = chunk["interval"]
            if (end - start) != chunk["chunk_size"]:
                raise Exception(
                    "The config intervals doesn't match the number of samples. This shouldn't have happened."
                )
            self._intervals.append(chunk["interval"])

        self._length = sum([chunk["chunk_size"] for chunk in self._chunks])
        self._downloader = Downloader(_remote_dir, cache_dir, self._chunks)

    @property
    def intervals(self):
        return self._intervals

    @property
    def data_format(self):
        return self._config["data_format"]

    @property
    def config(self):
        return self._config

    def _get_chunk_index_from_index(self, index: int) -> int:
        for chunk_index, internal in enumerate(self._intervals):
            if internal[0] <= index < internal[1]:
                return chunk_index
        raise ValueError(
            f"The provided index {index} didn't find a match within the chunk intervals {self._intervals}."
        )

    def __getitem__(self, index: ChunkedIndex) -> Tuple[str, int, int]:
        """Find the associated chunk metadata."""
        chunk = self._chunks[index.chunk_index]
        return os.path.join(self._cache_dir, chunk["filename"]), *self._intervals[index.chunk_index]

    @classmethod
    def load(cls, cache_dir: str, _remote_dir: Optional[str] = None) -> Optional["ChunksConfig"]:
        if isinstance(_remote_dir, str):
            Downloader.download_file_from_s3(
                os.path.join(_remote_dir, "index.json"), os.path.join(cache_dir, "index.json")
            )
        files = os.listdir(cache_dir)
        index_filenames = sorted([f for f in files if f.endswith("index.json")])
        if not index_filenames:
            return None
        return ChunksConfig(cache_dir, index_filenames, _remote_dir)

    def __len__(self) -> int:
        return self._length


class Downloader:
    def __init__(self, _remote_dir: str, cache_dir: str, chunks):
        self._processes = {}
        self._credentials = None
        self._remote_dir = _remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks

    def create_credentials(self):
        if self._credentials:
            return

        from botocore.credentials import InstanceMetadataProvider
        from botocore.utils import InstanceMetadataFetcher

        provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=1000, num_attempts=2))

        credentials = provider.load()

        os.environ["AWS_ACCESS_KEY"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token
        self._credentials = credentials

    def chunk_index_download(self, index: int) -> None:
        local_filepath = os.path.join(self._cache_dir, self._chunks[int(index)]["filename"])

        if os.path.exists(local_filepath):
            return

        self.create_credentials()

        remote_filepath = os.path.join(self._remote_dir, self._chunks[int(index)]["filename"])

        if remote_filepath.startswith("s3://"):
            self.download_file_from_s3_with_s5cmd(int(index), remote_filepath, local_filepath)

            return

        if remote_filepath.startswith("s3://"):
            self.download_file_from_s3(remote_filepath, local_filepath)

            return

        raise ValueError(f"The provided `remote_filepath` isn't supported. Found {remote_filepath}.")

    def download_file_from_s3_with_s5cmd(self, index, remote_filepath: str, local_filepath: str):
        if index not in self._processes:
            self._processes[index] = Popen(
                f"s5cmd cp {remote_filepath} {local_filepath}".split(" "),
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
            ).wait()

    @classmethod
    def download_file_from_s3(cls, remote_filepath: str, local_filepath: str):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        extra_args = {}

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
