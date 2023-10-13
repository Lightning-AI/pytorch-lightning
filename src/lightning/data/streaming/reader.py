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
from threading import Lock, Thread
from time import sleep
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv
from lightning.data.streaming.config import ChunksConfig
from lightning.data.streaming.constants import _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.serializers import _SERIALIZERS, Serializer

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import PyTree, tree_unflatten


class PrepareChunksThread(Thread):
    """This thread is responsible to download the chunks associated to a given worker."""

    def __init__(self, config: ChunksConfig) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._chunks_index_to_be_processed: List[int] = []
        self._chunks_index_to_ready: List[int] = []
        self._lock = Lock()

    def add(self, chunk_indices: List[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        with self._lock:
            self._chunks_index_to_be_processed.extend(chunk_indices)

    def run(self) -> None:
        while True:
            with self._lock:
                if len(self._chunks_index_to_be_processed) == 0:
                    sleep(0.007)
                    continue

                chunk_index = self._chunks_index_to_be_processed.pop(0)

            # TODO: Implement eviction
            self._config.download_chunk_from_index(chunk_index)
            self._chunks_index_to_ready.append(chunk_index)


class BinaryReader:
    def __init__(
        self,
        cache_dir: str,
        remote_dir: Optional[str] = None,
        compression: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[Union[int, Literal["latest"]]] = "latest",
    ) -> None:
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Arguments:
            cache_dir: The path to cache folder.
            remote_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.
            compression: The algorithm to decompress the chunks.
            name: The name of dataset in the cloud.
            version: The version of the dataset in the cloud to use. By default, we will use the latest.

        """
        super().__init__()
        self._cache_dir = cache_dir
        self._remote_dir = remote_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._intervals: Optional[List[str]] = None

        self._serializers: Dict[str, Serializer] = _SERIALIZERS
        self._distributed_env = _DistributedEnv.detect()
        self._rank: Optional[int] = None
        self._config: Optional[ChunksConfig] = None
        self._prepare_thread: Optional[PrepareChunksThread] = None

    def _get_chunk_index_from_index(self, index: int) -> int:
        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self._config._get_chunk_index_from_index(index)  # type: ignore

    def _try_load_config(self) -> Optional[ChunksConfig]:
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(self._cache_dir, self._remote_dir)
        return self._config

    @property
    def config(self) -> ChunksConfig:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config

    @property
    def rank(self) -> int:
        """Returns the rank of the writer."""
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    def read(self, index: ChunkedIndex) -> Any:
        """Read an item for the given from a chunk.

        If the chunk isn't available locally or in memory, it will be downloaded.

        Prefetching should reduce the wait time to be the batch available.

        """
        if not isinstance(index, ChunkedIndex):
            raise ValueError("The Reader.read(...) method expects a chunked Index.")

        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        # Create and start the prepare chunks thread
        if index.chunk_indexes is not None and self._prepare_thread is None and self._config:
            self._prepare_thread = PrepareChunksThread(self._config)
            self._prepare_thread.start()
            self._prepare_thread.add(index.chunk_indexes)

        # Fetch the element
        chunk_filepath, begin, _ = self.config[index]
        raw_item_data = self.load_item_from_chunk(index.index, chunk_filepath, begin)
        return self.deserialize(raw_item_data)

    def deserialize(self, raw_item_data: bytes) -> "PyTree":
        """Deserialize the raw bytes into their python equivalent."""
        idx = len(self.config.data_format) * 4
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self.config.data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self.config.config["data_spec"])

    def load_item_from_chunk(self, index: int, chunk_filepath: str, begin: int) -> bytes:
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
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return len(self.config)

    def get_chunk_interval(self) -> List[Tuple[int, int]]:
        """Get the index interval of each chunk."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self.config.intervals
