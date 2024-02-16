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
from logging import Logger
from queue import Empty
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

from lightning.data.constants import _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.config import ChunksConfig
from lightning.data.streaming.item_loader import BaseItemLoader, PyTreeLoader
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.serializers import Serializer, _get_serializers
from lightning.data.utilities.env import _DistributedEnv, _WorkerEnv

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

if _TORCH_GREATER_EQUAL_2_1_0:
    pass


logger = Logger(__name__)


_END_TOKEN = "END"

# Note: The timeout here should not be too short. We need to prevent the caller from aggressively
# querying the queue and consuming too many CPU cycles.
_DEFAULT_TIMEOUT = 0.1
_LONG_DEFAULT_TIMEOUT = 5


class PrepareChunksThread(Thread):
    """This thread is responsible to download the chunks associated to a given worker."""

    def __init__(
        self,
        config: ChunksConfig,
        item_loader: BaseItemLoader,
        distributed_env: _DistributedEnv,
        max_cache_size: Optional[int] = None,
        max_pre_download: int = 2,
    ) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._item_loader = item_loader
        self._max_pre_download = max_pre_download
        self._pre_download_counter = 0
        self._distributed_env = distributed_env

        self._chunks_index_to_be_deleted: List[int] = []
        self._max_cache_size = max_cache_size
        self._parent_cache_dir = os.path.dirname(self._config._cache_dir)
        self._to_download_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._to_delete_queue: multiprocessing.Queue = multiprocessing.Queue()

        # Check whether a dataset slice fits on the node
        num_bytes_per_nodes = self._config.num_bytes // self._distributed_env.num_nodes
        self._delete_chunks_when_processed = num_bytes_per_nodes > max_cache_size if max_cache_size else False
        self._has_exited = False

    def download(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_download_queue.put(chunk_index)

    def delete(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to delete for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_delete_queue.put(chunk_index)

    def _delete(self, chunk_index: int) -> None:
        """Inform the item loader of the chunk to delete."""
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.delete(chunk_index, chunk_filepath)

    def stop(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        self._to_download_queue.put(_END_TOKEN)

    def _maybe_delete_chunks(self) -> None:
        reached_pre_download = self._pre_download_counter == self._max_pre_download

        # we have already pre-downloaded some chunks, we just need to wait for them to be processed.
        chunk_index = _get_from_queue(
            self._to_delete_queue, timeout=_LONG_DEFAULT_TIMEOUT if reached_pre_download else _DEFAULT_TIMEOUT
        )

        if chunk_index is not None:
            self._pre_download_counter -= 1

            # Store the current chunk index
            self._chunks_index_to_be_deleted.append(chunk_index)

        # Get the current cache size and decide whether we need to start cleanup. Otherwise, keep track of it
        while self._max_cache_size and self._chunks_index_to_be_deleted and self._can_delete_chunk():
            # Delete the oldest chunk
            self._delete(self._chunks_index_to_be_deleted.pop(0))

        return

    def _can_delete_chunk(self) -> bool:
        if self._delete_chunks_when_processed:
            return self._pre_download_counter >= self._max_pre_download - 1
        return self._max_cache_size is not None and _get_folder_size(self._parent_cache_dir) >= self._max_cache_size

    def _pre_load_chunk(self, chunk_index: int) -> None:
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.pre_load_chunk(chunk_index, chunk_filepath)

    def run(self) -> None:
        while True:
            if self._pre_download_counter < self._max_pre_download:
                chunk_index = _get_from_queue(self._to_download_queue)
                if chunk_index == _END_TOKEN:
                    self._has_exited = True
                    return

                if chunk_index is not None:
                    self._config.download_chunk_from_index(chunk_index)

                    # Preload item if possible to gain some time but only
                    # if this is one of the pre-downloaded chunk
                    if self._pre_download_counter > 0:
                        self._pre_load_chunk(chunk_index)

                    # Avoid downloading too many chunks in advance at the risk of over using the disk space
                    self._pre_download_counter += 1

            if self._max_cache_size:
                self._maybe_delete_chunks()


class BinaryReader:
    def __init__(
        self,
        cache_dir: str,
        max_cache_size: Optional[Union[int, str]] = None,
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
        self._max_cache_size = int(os.getenv("MAX_CACHE_SIZE", max_cache_size or 0))

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
                self._prepare_thread = PrepareChunksThread(
                    self._config, self._item_loader, self._distributed_env, self._max_cache_size
                )
                self._prepare_thread.start()
                if index.chunk_indexes:
                    self._prepare_thread.download(index.chunk_indexes)

            # If the chunk_index is new, request for it to be downloaded.
            if index.chunk_index != self._last_chunk_index:
                assert self._prepare_thread
                self._prepare_thread.download([index.chunk_index])

            if self._last_chunk_index is None:
                self._last_chunk_index = index.chunk_index

        # Fetch the element
        chunk_filepath, begin, _ = self.config[index]
        item = self._item_loader.load_item_from_chunk(index.index, index.chunk_index, chunk_filepath, begin)

        # We need to request deletion after the latest element has been loaded.
        # Otherwise, this could trigger segmentation fault error depending on the item loader used.
        if self._config and self._config._remote_dir and index.chunk_index != self._last_chunk_index:
            assert self._prepare_thread
            assert self._last_chunk_index is not None

            # inform the chunk has been completely consumed
            self._prepare_thread.delete([self._last_chunk_index])

            # track the new chunk index as the latest one
            self._last_chunk_index = index.chunk_index

        if index.is_last_index and self._prepare_thread:
            # inform the thread it is time to stop
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


def _get_folder_size(path: str) -> int:
    """Collect the size of each files within a folder.

    This method is robust to file deletion races

    """
    size = 0
    for dirpath, _, filenames in os.walk(str(path)):
        for filename in filenames:
            with contextlib.suppress(FileNotFoundError):
                size += os.stat(os.path.join(dirpath, filename)).st_size
    return size


def _get_from_queue(queue: multiprocessing.Queue, timeout: float = _DEFAULT_TIMEOUT) -> Optional[Any]:
    try:
        return queue.get(timeout=timeout)
    except Empty:
        pass
    except OSError as err:
        # handle closed queue before the thread terminates
        if "handle is closed" in str(err) or "Bad file descriptor" in str(err):
            logger.debug(err)
        else:
            raise err
    except EOFError as err:
        logger.debug(err)
    return None
