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
from contextlib import contextmanager
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Dict, Optional

import numpy as np

from lightning.data.cache.config import ChunksConfig
from lightning.data.cache.pytree import tree_unflatten
from lightning.data.cache.sampler import ChunkedIndex
from lightning.data.cache.serializers import _SERIALIZERS, Serializer
from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv


class PrepareChunksThread(Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self.chunks_index_to_be_processed = []
        self.chunks_index_to_ready = []
        self.lock = Lock()

    def add(self, chunk_indices):
        with self.lock:
            self.chunks_index_to_be_processed.extend(chunk_indices)

    def run(self):
        while True:
            with self.lock:
                if len(self.chunks_index_to_be_processed) == 0:
                    sleep(0.007)
                    continue
                chunk_index = self.chunks_index_to_be_processed.pop(0)
            self.config._downloader.chunk_index_download(chunk_index)


class BinaryReader:
    def __init__(self, cache_dir: str, remote_dir: Optional[str] = None, compression: Optional[str] = None):
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Arguments:
            cache_dir: The path to cache folder.
            remote_dir: The path to a remote folder where the data are located.
            compression: The algorithm to decompress the chunks.

        """
        super().__init__()
        self._cache_dir = cache_dir
        self._remote_dir = remote_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._config = None
        self._intervals = None

        self._chunks_data = {}
        self._serializers: Dict[str, Serializer] = _SERIALIZERS

        self._distributed_env = _DistributedEnv.detect()
        self._rank = None
        self._config: Optional[ChunksConfig] = None
        self._latest_chunk_index = None
        self._executor = None
        self._prepare_thread = None

    def _try_load_config(self):
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(self._cache_dir, self._remote_dir)

    @property
    def rank(self):
        """Returns the rank of the writer."""
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    @contextmanager
    def measure_on_rank_0(self, msg: str):
        if self.rank == 0:
            t0 = time()
            yield
            print(msg, time() - t0)

    def read(self, index: ChunkedIndex):
        """Read an item for the given from a chunk.

        If the chunk isn't available locally or in memory, it will be downloaded.

        Prefetching should reduce the wait time to be the batch available.

        """
        if not isinstance(index, ChunkedIndex):
            raise ValueError("The Reader.read(...) method expects a chunked Index.")

        # Load the config containing the index
        if self._config is None:
            self._try_load_config()

            if self._config is None:
                raise Exception("The reader index isn't defined.")

        # Create and start the prepare chunks thread
        if self._prepare_thread is None:
            self._prepare_thread = PrepareChunksThread(self._config)
            self._prepare_thread.start()

        # Register the chunks to be downloaded
        if index.chunk_indexes is not None:
            self._prepare_thread.add(index.chunk_indexes)

        # Fetch the element
        chunk_filepath, begin, end = self._config[index]
        raw_item_data = self.load_item_from_chunk(index.index, chunk_filepath, begin)
        return self.deserialize(raw_item_data)

    def deserialize(self, raw_item_data: bytes) -> Any:
        """Deserialize the raw bytes into their python equivalent."""
        idx = len(self._config.data_format) * 4
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self._config.data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config.config["data_spec"])

    def load_item_from_chunk(self, index: int, chunk_filepath: str, begin: int):
        offset = (1 + (index - begin)) * 4

        while not os.path.exists(chunk_filepath):
            sleep(0.0001)

        with open(chunk_filepath, "rb", 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
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
