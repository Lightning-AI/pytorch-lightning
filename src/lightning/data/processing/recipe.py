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
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from time import sleep
from typing import Any, List, Optional, TypeVar, Union
from urllib import parse

import botocore

from lightning.data.constants import (
    _INDEX_FILENAME,
    _TORCH_GREATER_EQUAL_2_1_0,
)
from lightning.data.streaming import Cache
from lightning.data.streaming.cache import Dir
from lightning.data.streaming.client import S3Client
from lightning.data.utilities.env import _get_cache_dir, _get_home_folder, _get_node_rank, _get_num_nodes

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import tree_unflatten, treespec_loads


@dataclass
class _Result:
    size: Optional[int] = None
    num_bytes: Optional[str] = None
    data_format: Optional[str] = None
    compression: Optional[str] = None
    num_chunks: Optional[int] = None
    num_bytes_per_chunk: Optional[List[int]] = None


T = TypeVar("T")


class DataRecipe:
    @abstractmethod
    def prepare_structure(self, input_dir: Optional[str]) -> List[T]:
        pass

    @abstractmethod
    def prepare_item(self, *args: Any) -> Any:
        pass

    def listdir(self, path: str) -> List[str]:
        home = _get_home_folder()
        filepath = os.path.join(home, ".cache", f"{self._name}/filepaths.txt")

        if os.path.exists(filepath):
            lines = []
            with open(filepath) as f:
                for line in f.readlines():
                    lines.append(line.replace("\n", ""))
            return lines

        filepaths = []
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepaths.append(os.path.join(dirpath, filename))

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            for filepath in filepaths:
                f.write(f"{filepath}\n")

        return filepaths

    def __init__(self) -> None:
        self._name: Optional[str] = None

    def _done(self, size: int, delete_cached_files: bool, output_dir: Dir) -> _Result:
        return _Result(size=size)


class DataChunkRecipe(DataRecipe):
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[Union[int, str]] = None,
        compression: Optional[str] = None,
    ):
        super().__init__()
        if chunk_size is not None and chunk_bytes is not None:
            raise ValueError("Either one of the `chunk_size` or the `chunk_bytes` need to be provided.")

        self.chunk_size = chunk_size
        self.chunk_bytes = 1 << 26 if chunk_size is None else chunk_bytes
        self.compression = compression

    @abstractmethod
    def prepare_structure(self, input_dir: Optional[str]) -> List[T]:
        """Return the structure of your data.

        Each element should contain at least a filepath.

        """

    @abstractmethod
    def prepare_item(self, item_metadata: T) -> Any:  # type: ignore
        """The return of this `prepare_item` method is persisted in chunked binary files."""

    def _done(self, size: int, delete_cached_files: bool, output_dir: Dir) -> _Result:
        num_nodes = _get_num_nodes()
        cache_dir = _get_cache_dir()

        chunks = [file for file in os.listdir(cache_dir) if file.endswith(".bin")]
        if chunks and delete_cached_files and output_dir.path is not None:
            raise RuntimeError(f"All the chunks should have been deleted. Found {chunks}")

        merge_cache = Cache(cache_dir, chunk_bytes=1)
        node_rank = _get_node_rank()
        merge_cache._merge_no_wait(node_rank if num_nodes > 1 else None)
        self._upload_index(output_dir, cache_dir, num_nodes, node_rank)

        if num_nodes == node_rank + 1:
            with open(os.path.join(cache_dir, _INDEX_FILENAME)) as f:
                config = json.load(f)

            size = sum([c["dim"] if c["dim"] is not None else c["chunk_size"] for c in config["chunks"]])
            num_bytes = sum([c["chunk_bytes"] for c in config["chunks"]])
            data_format = tree_unflatten(config["config"]["data_format"], treespec_loads(config["config"]["data_spec"]))

            return _Result(
                size=size,
                num_bytes=num_bytes,
                data_format=data_format,
                compression=config["config"]["compression"],
                num_chunks=len(config["chunks"]),
                num_bytes_per_chunk=[c["chunk_size"] for c in config["chunks"]],
            )
        return _Result(
            size=size,
        )

    def _upload_index(self, output_dir: Dir, cache_dir: str, num_nodes: int, node_rank: Optional[int]) -> None:
        """This method upload the index file to the remote cloud directory."""
        if output_dir.path is None and output_dir.url is None:
            return

        obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)
        if num_nodes > 1:
            local_filepath = os.path.join(cache_dir, f"{node_rank}-{_INDEX_FILENAME}")
        else:
            local_filepath = os.path.join(cache_dir, _INDEX_FILENAME)

        if obj.scheme == "s3":
            s3 = S3Client()
            s3.client.upload_file(
                local_filepath, obj.netloc, os.path.join(obj.path.lstrip("/"), os.path.basename(local_filepath))
            )
        elif os.path.isdir(output_dir.path):
            shutil.copyfile(local_filepath, os.path.join(output_dir.path, os.path.basename(local_filepath)))

        if num_nodes == 1 or node_rank is None:
            return

        # Merge the index files generated by each node.
        # Note: When using the Data Optimizer, they should be a single process on each node executing this section
        # So no risk to get race conditon.
        if num_nodes == node_rank + 1:
            # Get the index file locally
            for node_rank in range(num_nodes - 1):
                remote_filepath = os.path.join(
                    output_dir.url if output_dir.url else output_dir.path, f"{node_rank}-{_INDEX_FILENAME}"
                )
                node_index_filepath = os.path.join(cache_dir, os.path.basename(remote_filepath))
                if obj.scheme == "s3":
                    obj = parse.urlparse(remote_filepath)
                    _wait_for_file_to_exist(s3, obj)
                    with open(node_index_filepath, "wb") as f:
                        s3.client.download_fileobj(obj.netloc, obj.path.lstrip("/"), f)
                elif os.path.isdir(output_dir.path):
                    shutil.copyfile(remote_filepath, node_index_filepath)

            merge_cache = Cache(cache_dir, chunk_bytes=1)
            merge_cache._merge_no_wait()
            self._upload_index(output_dir, cache_dir, 1, None)


class DataTransformRecipe(DataRecipe):
    @abstractmethod
    def prepare_structure(self, input_dir: Optional[str]) -> List[T]:
        """Return the structure of your data.

        Each element should contain at least a filepath.

        """

    @abstractmethod
    def prepare_item(self, output_dir: str, item_metadata: T) -> None:  # type: ignore
        """Use your item metadata to process your files and save the file outputs into `output_dir`."""


def _wait_for_file_to_exist(s3: S3Client, obj: parse.ParseResult, sleep_time: int = 2) -> Any:
    """This function check."""
    while True:
        try:
            return s3.client.head_object(Bucket=obj.netloc, Key=obj.path.lstrip("/"))
        except botocore.exceptions.ClientError as e:
            if "the HeadObject operation: Not Found" in str(e):
                sleep(sleep_time)
            else:
                raise e
