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
import warnings
from threading import Lock, Thread
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

from lightning.data.streaming.config import ChunksConfig
from lightning.data.streaming.constants import _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.item_loader import BaseItemLoader, PyTreeLoader
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.serializers import Serializer, _get_serializers
from lightning.data.utilities.env import _DistributedEnv, _WorkerEnv

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

if _TORCH_GREATER_EQUAL_2_1_0:
    pass


class PrepareChunksThread(Thread):
    """This thread is responsible to download the chunks associated to a given worker."""

    def __init__(self, config: ChunksConfig, max_cache_size: Optional[int] = None, pre_download: int = 10) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._chunks_index_to_be_downloaded: List[int] = []
        self._chunks_index_to_be_deleted: List[int] = []
        self._lock = Lock()
        self._max_cache_size = max_cache_size
        self._downloaded_chunks = 0
        self._processed_chunks = 0
        self._processed_chunks_counter = 0
        self._delete_chunks = 0
        self._pre_download = pre_download

    def download(self, chunk_indices: List[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        with self._lock:
            for chunk_indice in chunk_indices:
                if chunk_indice not in self._chunks_index_to_be_downloaded:
                    self._chunks_index_to_be_downloaded.append(chunk_indice)

    def delete(self, chunk_indices: List[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        with self._lock:
            for chunk_indice in chunk_indices:
                if chunk_indice not in self._chunks_index_to_be_deleted:
                    self._chunks_index_to_be_deleted.append(chunk_indice)
                    self._processed_chunks += 1
                    self._processed_chunks_counter += 1

    def _delete(self, chunk_index: int) -> None:
        chunk_filepath, begin, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]

        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)

    def run(self) -> None:
        while True:
            with self._lock:
                # Wait for something to do
                if len(self._chunks_index_to_be_downloaded) == 0 and len(self._chunks_index_to_be_deleted) == 0:
                    sleep(0.01)
                    continue

                # Delete the chunks if we are missing disk space.
                if self._max_cache_size and self._processed_chunks_counter >= self._pre_download:
                    if shutil.disk_usage(self._config._cache_dir).total >= self._max_cache_size:
                        for chunk_index in self._chunks_index_to_be_deleted:
                            if chunk_index not in self._chunks_index_to_be_downloaded:
                                self._delete(chunk_index)
                                self._delete_chunks += 1
                                self._processed_chunks_counter = 0
                    self._chunks_index_to_be_deleted = []

                # If there is no chunks to download, go back to waiting
                if len(self._chunks_index_to_be_downloaded) == 0:
                    continue

                # If we have already downloaded too many chunks, let's wait for processed chunks to catch up
                if self._max_cache_size and (self._downloaded_chunks - self._processed_chunks) > self._pre_download:
                    sleep(0.01)
                    continue

                chunk_index = self._chunks_index_to_be_downloaded.pop(0)

            self._config.download_chunk_from_index(chunk_index)
            self._downloaded_chunks += 1


class BinaryReader:
    def __init__(
        self,
        cache_dir: str,
        max_cache_size: int,
        remote_input_dir: Optional[str] = None,
        compression: Optional[str] = None,
        item_loader: Optional[BaseItemLoader] = None,
        serializers: Optional[Dict[str, Serializer]] = None,
    ) -> None:
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Arguments:
            cache_dir: The path to cache folder.
            remote_input_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.
            compression: The algorithm to decompress the chunks.
            item_loader: The chunk sampler to create sub arrays from a chunk.
            max_cache_size: The maximum cache size used by the reader when fetching the chunks.
            serializers: Provide your own serializers.

        """
        super().__init__()
        warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

        self._cache_dir = cache_dir
        self._remote_input_dir = remote_input_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._intervals: Optional[List[str]] = None

        self._serializers: Dict[str, Serializer] = _get_serializers(serializers)
        self._distributed_env = _DistributedEnv.detect()
        self._rank: Optional[int] = None
        self._config: Optional[ChunksConfig] = None
        self._prepare_thread: Optional[PrepareChunksThread] = None
        self._chunks_index_to_be_downloaded: List[int] = []
        self._item_loader = item_loader or PyTreeLoader()
        self._last_chunk_index: Optional[int] = None
        self._max_cache_size = int(os.getenv("MAX_CACHE_SIZE", max_cache_size))

    def _get_chunk_index_from_index(self, index: int) -> int:
        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self._config._get_chunk_index_from_index(index)  # type: ignore

    def _try_load_config(self) -> Optional[ChunksConfig]:
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(self._cache_dir, self._serializers, self._remote_input_dir, self._item_loader)
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

        if self._config and self._config._remote_dir:
            # Create and start the prepare chunks thread
            if self._prepare_thread is None and self._config:
                self._prepare_thread = PrepareChunksThread(self._config, self._max_cache_size)
                self._prepare_thread.start()
                if index.chunk_indexes:
                    self._chunks_index_to_be_downloaded.extend(index.chunk_indexes)
                    self._prepare_thread.download(index.chunk_indexes)

            # If the chunk_index isn't already in the download and delete queues, add it.
            if index.chunk_index != self._last_chunk_index:
                assert self._prepare_thread

                if self._last_chunk_index:
                    self._prepare_thread.delete([self._last_chunk_index])

                self._last_chunk_index = index.chunk_index
                self._prepare_thread.download([index.chunk_index])

        # Fetch the element
        chunk_filepath, begin, _ = self.config[index]
        return self._item_loader.load_item_from_chunk(index.index, index.chunk_index, chunk_filepath, begin)

    def get_length(self) -> int:
        """Get the number of samples across all chunks."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return len(self.config)

    def get_chunk_intervals(self) -> List[Tuple[int, int]]:
        """Get the index interval of each chunk."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self.config.intervals

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_prepare_thread"] = None
        return state
