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
from typing import Optional

import numpy as np

from lightning.data.cache.serializers import _SERIALIZERS


class BinaryReader:
    def __init__(self, _cache_dir: str, compression: Optional[str] = None):
        super().__init__()
        self._cache_dir = _cache_dir
        self._compression = compression
        self._index = None
        self._intervals = None
        self._chunks_data = {}
        self._serializers = _SERIALIZERS

    def _try_read_index(self):
        files = os.listdir(self._cache_dir)
        indexes_filepath = sorted([os.path.join(self._cache_dir, f) for f in files if f.endswith("index.json")])
        if not indexes_filepath:
            return

        index = {"chunks": []}
        for path in indexes_filepath:
            with open(path) as f:
                data = json.load(f)
                index["chunks"].extend(data["chunks"])

        self._index = index

        for chunk in self._index["chunks"]:
            chunk["data"] = None
            self._chunks_data[chunk["filename"]] = chunk

        self._intervals = []
        cumsum_samples = np.cumsum([0] + [v["samples"] for v in self._index["chunks"]] + [1])
        for i in range(len(cumsum_samples) - 1):
            self._intervals.append([cumsum_samples[i], cumsum_samples[i + 1]])

    def _map_index_to_chunk_id(self, index):
        for interval_index, internal in enumerate(self._intervals):
            if internal[0] <= index and index < internal[1]:
                return interval_index
        return None

    def _should_keep_in_memory(self):
        return True

    def read(self, index: int, rank: int = 0):
        if self._index is None:
            self._try_read_index()

        if self._index is None:
            raise Exception("The reader index isn't defined.")

        chunk_id = self._map_index_to_chunk_id(index)
        chunk_config = self._index["chunks"][chunk_id]
        chunk_path = os.path.join(self._cache_dir, chunk_config["filename"])
        raw_item_data, item_config = self.load_item_from_chunk(
            index, chunk_path, keep_in_memory=self._should_keep_in_memory()
        )
        return self.deserialize(raw_item_data, item_config)

    def deserialize(self, raw_item_data, item_config):
        sizes = []
        idx = 0
        data_format = item_config["data_format"]
        keys = sorted(data_format)
        for key in keys:
            (size,) = np.frombuffer(raw_item_data[idx : idx + 4], np.uint32)
            sizes.append(size)
            idx += 4
        sample = {}
        for key, size in zip(keys, sizes):
            value = raw_item_data[idx : idx + size]
            serializer = self._serializers[data_format[key]]
            sample[key] = serializer.deserialize(value)
            idx += size
        return sample

    def load_item_from_chunk(self, index: int, chunk_path: str, keep_in_memory: bool = False):
        chunk_name = os.path.basename(chunk_path)
        begin, end = self._chunks_data[chunk_name]["mapping"][str(index)]
        config = self._chunks_data[chunk_name]["config"]
        if self._chunks_data[chunk_name]["data"] is not None:
            return self._chunks_data[chunk_name]["data"][begin:end], config

        if keep_in_memory:
            with open(chunk_path, "rb", 0) as fp:
                data = fp.read()
            self._chunks_data[chunk_name]["data"] = data
            return data[begin:end], config

        with open(chunk_path, "rb", 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data, config

    def get_length(self) -> int:
        if self._index is None:
            self._try_read_index()

        if self._index is None:
            raise Exception("The reader index isn't defined.")

        return sum([v["samples"] for v in self._index["chunks"]])

    def get_chunk_interval(self):
        if self._index is None:
            self._try_read_index()

        if self._intervals is None:
            raise Exception("The reader index isn't defined.")

        return self._intervals
