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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from lightning.data.cache.pytree import tree_flatten, tree_unflatten, treespec_loads
from lightning.data.cache.serializers import _SERIALIZERS, Serializer
from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv


class ChunksConfig:
    def __init__(self, cache_dir: str, index_filenames: str):
        self._cache_dir = cache_dir
        self.index_filenames = sorted(index_filenames)
        self._intervals = []
        self._config = None
        self._chunks = []

        for filename in self.index_filenames:
            with open(os.path.join(self._cache_dir, filename)) as f:
                data = json.load(f)

                if self._config is None:
                    self._config = data["config"]
                    self._config["data_spec"] = treespec_loads(self._config["data_spec"])
                    flattened_data_format, _ = tree_flatten(self._config["data_format"])
                    self._config["flattened_data_format"] = flattened_data_format

                elif self._config != data["config"]:
                    raise Exception("The config isn't consistent between chunks. This shouldn't have happened.")

                self._chunks.extend(data["chunks"])

        for chunk in self._chunks:
            start, end = chunk["interval"]
            if (end - start + 1) != chunk["samples"]:
                raise Exception(
                    "The config intervals doesn't match the number of samples. This shouldn't have happened."
                )
            self._intervals.append(chunk["interval"])

    @property
    def intervals(self):
        return self._intervals

    @property
    def config(self):
        return self._config

    def __getitem__(self, index: int) -> Tuple[str, int, int]:
        """Find the associated chunk metadata."""
        for interval_index, internal in enumerate(self._intervals):
            if internal[0] <= index and index <= internal[1]:
                chunk = self._chunks[interval_index]
                mapping = chunk["mapping"][str(index)]
                return os.path.join(self._cache_dir, chunk["filename"]), *mapping
        raise Exception(f"The chunk interval weren't properly defined. Found {self._intervals} for index {index}.")

    @classmethod
    def load(cls, cache_dir: str) -> Optional["ChunksConfig"]:
        files = os.listdir(cache_dir)
        index_filenames = sorted([f for f in files if f.endswith("index.json")])
        if not index_filenames:
            return None
        return ChunksConfig(cache_dir, index_filenames)


class BinaryReader:
    def __init__(self, cache_dir: str, compression: Optional[str] = None):
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Arguments:
            cache_dir: The path to cache folder
            compression: The algorithm to decompress the chunks.

        """

        super().__init__()
        self._cache_dir = cache_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._index = None
        self._intervals = None

        self._chunks_data = {}
        self._serializers: Dict[str, Serializer] = _SERIALIZERS

        self._env = _DistributedEnv.detect()
        self._worker_env: Optional[_WorkerEnv] = None

        self._config: Optional[ChunksConfig] = None

    def _try_load_config(self):
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(self._cache_dir)

    def read(self, index: int):
        """Read an item for the given from a chunk.

        If the chunk isn't available, it will be downloaded.

        """
        if self._index is None:
            self._try_load_config()

        if self._config is None:
            raise Exception("The reader index isn't defined.")

        chunk_filepath, begin, end = self._config[index]
        raw_item_data = self.load_item_from_chunk(chunk_filepath, begin, end, keep_in_memory=True)
        return self.deserialize(raw_item_data)

    def deserialize(self, raw_item_data: bytes) -> Any:
        """Deserialize the raw bytes into their python equivalent."""
        sizes = []
        idx = 0
        data_format = self._config.config["flattened_data_format"]
        for key in data_format:
            (size,) = np.frombuffer(raw_item_data[idx : idx + 4], np.uint32)
            sizes.append(size)
            idx += 4
        data = []
        for size, format in zip(sizes, data_format):
            serializer = self._serializers[format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config.config["data_spec"])

    def load_item_from_chunk(self, chunk_filepath: str, begin: int, end: int, keep_in_memory: bool = False):
        if chunk_filepath in self._chunks_data:
            return self._chunks_data[chunk_filepath][begin:end]

        if keep_in_memory:
            with open(chunk_filepath, "rb", 0) as fp:
                data = fp.read()
            self._chunks_data[chunk_filepath] = data
            return data[begin:end]

        with open(chunk_filepath, "rb", 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data

    def get_length(self) -> int:
        """Get the number of samples across all chunks."""
        if self._index is None:
            self._try_load_config()

        if self._index is None:
            raise Exception("The reader index isn't defined.")

        return sum([v["samples"] for v in self._index["chunks"]])

    def get_chunk_interval(self):
        """Get the index interval of each chunks."""
        if self._index is None:
            self._try_load_config()

        if self._intervals is None:
            raise Exception("The reader index isn't defined.")

        return self._intervals
