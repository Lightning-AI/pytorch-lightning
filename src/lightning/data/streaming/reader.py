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

import contextlib
import multiprocessing
import os
import warnings
from queue import Empty
from threading import Thread
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

    def __init__(self, config: ChunksConfig, item_loader, max_cache_size: Optional[int] = None, max_pre_download: int = 10) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._item_loader = item_loader
        self._chunks_index_to_be_downloaded: List[int] = []
        chunk_indexes = self._collect_ordered_chunk_indexes_from_cache()
        self._chunks_index_to_be_deleted: List[int] = []
        self._max_cache_size = max_cache_size
        self._parent_cache_dir = os.path.dirname(self._config._cache_dir)
        self._to_download_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._to_delete_queue: multiprocessing.Queue = multiprocessing.Queue()

        # populate back the queues with existing items. As they already exists, this is almost a no-op
        for chunk_index in chunk_indexes:
            self._to_download_queue.put(chunk_index)
            self._to_delete_queue.put(chunk_index)

        self._to_stop_queue: multiprocessing.Queue = multiprocessing.Queue()

        self._max_pre_download = max_pre_download
        self._pre_download_counter = 0

    def _collect_ordered_chunk_indexes_from_cache(self) -> List[int]:
        chunk_indexes = [
            [self._config._get_chunk_index_from_filename(f), os.path.getctime(os.path.join(self._config._cache_dir, f))]
            for f in os.listdir(self._config._cache_dir)
            if f.endswith(".bin")
        ]
        return [x[0] for x in sorted(chunk_indexes, key=lambda x: x[1])]

    def download(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_download_queue.put(chunk_index)

    def delete(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to delete for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_delete_queue.put(chunk_index)

    def _delete(self, chunk_index: int) -> None:
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.delete(chunk_index, chunk_filepath)

    def stop(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        self._to_stop_queue.put(None)

    def _maybe_delete_chunks(self):
        try:
            # Whether the reader has already finished processing a chunk
            chunk_index = self._to_delete_queue.get(timeout=0.01)
            self._pre_download_counter -= 1

            # Store the current chunk index
            self._chunks_index_to_be_deleted.append(chunk_index)

            # Get the current cache size and decide whether we need to start cleanup. Otherwise, keep track of it
            while self._chunks_index_to_be_deleted and _get_folder_size(self._parent_cache_dir) >= self._max_cache_size:
                self._delete(self._chunks_index_to_be_deleted.pop(0))
                print("DELETE AAA")
        except Empty:
            pass
        except OSError as e:
            # handle closed queue before the thread terminates
            if "handle is closed" in str(e):
                pass
            else:
                raise e

    def _maybe_flush_cache(self, chunk_index: int) -> None:
        # Before downloading, check whether we have enough space
        while self._max_cache_size and _get_folder_size(self._parent_cache_dir) >= self._max_cache_size:
            # Get chunk_filepath associated to this chunk_index
            chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
            if os.path.exists(chunk_filepath):
                break

            # delete the oldest file as we need the space
            has_deleted = _try_to_delete_oldest_chunk(self._config._cache_dir)

            # there were nothing to delete
            if not has_deleted:
                break

    def run(self) -> None:
        while True:
            try:
                if  self._pre_download_counter <= self._max_pre_download:
                    chunk_index = self._to_download_queue.get(timeout=0.01)
                    self._maybe_flush_cache(chunk_index)
                    self._config.download_chunk_from_index(chunk_index)

                    # Avoid downloading too many chunks in advance at the risk of over using the disk space
                    self._pre_download_counter += 1
            except Empty:
                pass
            except OSError as e:
                # handle closed queue before the thread terminates
                if "handle is closed" in str(e):
                    pass
                else:
                    raise e

            if self._max_cache_size:
                self._maybe_delete_chunks()

            try:
                self._to_stop_queue.get(timeout=0.01)
                return
            except Empty:
                pass
            except OSError as e:
                # handle closed queue before the thread terminates
                if "handle is closed" in str(e):
                    return
                raise e

            sleep(0.01)


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
                self._prepare_thread = PrepareChunksThread(self._config, self._item_loader, self._max_cache_size)
                self._prepare_thread.start()
                if index.chunk_indexes:
                    self._prepare_thread.download(index.chunk_indexes)

            # If the chunk_index is new, request for it to be downloaded.
            if index.chunk_index != self._last_chunk_index:
                assert self._prepare_thread
                self._prepare_thread.download([index.chunk_index])

        # Fetch the element
        chunk_filepath, begin, _ = self.config[index]
        item = self._item_loader.load_item_from_chunk(index.index, index.chunk_index, chunk_filepath, begin)

        if self._config and self._config._remote_dir and index.chunk_index != self._last_chunk_index:
            assert self._prepare_thread
            if self._last_chunk_index:
                self._prepare_thread.delete([self._last_chunk_index])
            self._last_chunk_index = index.chunk_index

        if index.is_last_index and self._prepare_thread:
            self._prepare_thread.stop()
            self._prepare_thread = None

        return item

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


def _try_to_delete_oldest_chunk(dir_path: str) -> bool:
    filepaths = []
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if not filename.endswith(".bin"):
                continue
            try:
                filepath = os.path.join(dirpath, filename)
                filepaths.append([filepath, os.path.getctime(filepath)])
            except FileNotFoundError:
                pass

    if not filepaths:
        return False

    filepaths = sorted(filepaths, key=lambda x: x[1])
    os.remove(filepaths[0][0])
    return True


def _get_folder_size(path: str) -> int:
    size = 0
    for dirpath, _, filenames in os.walk(str(path)):
        for filename in filenames:
            with contextlib.suppress(FileNotFoundError):
                size += os.stat(os.path.join(dirpath, filename)).st_size
    return size
