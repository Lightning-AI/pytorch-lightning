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


class Reader:
    def __init__(self, _cache_dir: str, compression: Optional[str] = None):
        super().__init__()

        self._cache_dir = _cache_dir
        self._compression = compression
        self._index = None
        self._intervals = None
        self._chunks = []

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

        self._intervals = []
        cumsum_samples = np.cumsum([0] + [v["samples"] for v in self._index["chunks"]] + [1])
        for i in range(len(cumsum_samples) - 1):
            self._intervals.append([cumsum_samples[i], cumsum_samples[i + 1]])

        print(self._intervals)

    def _map_index_to_chunk_id(self, index):
        for interval_index, internal in enumerate(self._intervals):
            print(internal, index)
            if internal[0] <= index and index < internal[1]:
                return interval_index
        return None

    def read(self, index: int, rank):
        if self._index is None:
            self._try_read_index()

        if self._index is None:
            raise Exception("The reader index isn't defined.")

        chunk_id = self._map_index_to_chunk_id(index)
        chunk_config = self._index["chunks"][chunk_id]
        chunk_path = os.path.join(self._cache_dir, chunk_config["filename"])
        if not os.path.exists(chunk_path):
            download_chunk(chunk_path)

        return self.load_data_from_chunk(chunk_path)

    def load_data_from_chunk(self, chunk_path):
        pass

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
