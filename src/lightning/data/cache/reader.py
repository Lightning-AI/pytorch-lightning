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
from typing import Any, Dict, Optional, Tuple, Union
from urllib import parse
import numpy as np

from lightning.data.cache.pytree import tree_unflatten, treespec_loads
from lightning.data.cache.sampler import BatchIndex
from lightning.data.cache.serializers import _SERIALIZERS, Serializer
from lightning.data.datasets.env import _DistributedEnv
from lightning.data.cache.config import ChunksConfig


class BinaryReader:
    def __init__(self, cache_dir: str, source_dir: Optional[str] = None, compression: Optional[str] = None):
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Arguments:
            cache_dir: The path to cache folder
            compression: The algorithm to decompress the chunks.

        """

        super().__init__()
        self._cache_dir = cache_dir
        self._source_dir = source_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._config = None
        self._intervals = None

        self._chunks_data = {}
        self._serializers: Dict[str, Serializer] = _SERIALIZERS

        self._env = _DistributedEnv.detect()
        self._config: Optional[ChunksConfig] = None

    def _try_load_config(self):
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(self._cache_dir, self._source_dir)

    def read(self, index: Union[int, BatchIndex]):
        """Read an item for the given from a chunk.

        If the chunk isn't available locally or in memory, it will be downloaded.

        Prefetching should reduce the wait time to be the batch available.

        """
        if self._config is None:
            self._try_load_config()

        if self._config is None:
            raise Exception("The reader index isn't defined.")

        chunk_filepath, begin, end = self._config[index]
        raw_item_data = self.load_item_from_chunk(chunk_filepath, begin, end, keep_in_memory=False)
        return self.deserialize(raw_item_data)

    def deserialize(self, raw_item_data: bytes) -> Any:
        """Deserialize the raw bytes into their python equivalent."""
        sizes = []
        idx = 0
        data_format = self._config.data_format
        for _ in data_format:
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
            with open(chunk_filepath, "rb") as fp:
                fp.seek(0)
                data = fp.read()
            self._chunks_data[chunk_filepath] = data
            return data[begin:end]

        with open(chunk_filepath, "rb", 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data

    def get_length(self) -> int:
        """Get the number of samples across all chunks."""
        if self._config is None:
            self._try_load_config()

        if self._config is None:
            raise Exception("The reader index isn't defined.")

        return len(self._config)

    def get_chunk_interval(self):
        """Get the index interval of each chunk."""
        if self._config is None:
            self._try_load_config()

        if self._config is None:
            raise Exception("The reader index isn't defined.")

        return self._config.intervals