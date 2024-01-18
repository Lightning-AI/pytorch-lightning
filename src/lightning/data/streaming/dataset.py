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

import hashlib
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from lightning.data.streaming import Cache
from lightning.data.streaming.constants import (
    _DEFAULT_CACHE_DIR,
    _INDEX_FILENAME,
)
from lightning.data.streaming.item_loader import BaseItemLoader
from lightning.data.streaming.resolver import Dir, _resolve_dir
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.serializers import Serializer
from lightning.data.streaming.shuffle import FullShuffle, NoShuffle, Shuffle
from lightning.data.utilities.env import Environment, _DistributedEnv, _WorkerEnv


class StreamingDataset(IterableDataset):
    """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class."""

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        item_loader: Optional[BaseItemLoader] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        serializers: Optional[Dict[str, Serializer]] = None,
        max_cache_size: Union[int, str] = "100GB",
    ) -> None:
        """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class.

        Arguments:
            input_dir: Path to the folder where the input data is stored.
            item_loader: The logic to load an item from a chunk.
            shuffle: Whether to shuffle the data.
            drop_last: If `True`, drops the last items to ensure that
                all processes/workers return the same amount of data.
            seed: Random seed for shuffling.
            serializers: The serializers used to serialize and deserialize the chunks.
            max_cache_size: The maximum cache size used by the StreamingDataset.

        """
        super().__init__()
        if not isinstance(shuffle, bool):
            raise ValueError(f"Shuffle should be a boolean. Found {shuffle}")

        input_dir = _resolve_dir(input_dir)

        self.input_dir = input_dir

        self.item_loader = item_loader
        self.shuffle: bool = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.max_cache_size = max_cache_size

        self.cache: Optional[Cache] = None
        self.distributed_env = _DistributedEnv.detect()
        self.worker_env: Optional[_WorkerEnv] = None
        self.worker_chunks: List[int] = []
        self.worker_intervals: List[List[int]] = []
        self.current_indexes: List[int] = []
        self.chunk_index = 0
        self.num_chunks: Optional[int] = None
        self.global_index = 0
        self.index = 0
        self.has_triggered_download = False
        self.min_items_per_replica: Optional[int] = None
        self.current_epoch = 1
        self.random_state = None
        self.shuffler: Optional[Shuffle] = None
        self.serializers = serializers
        self._state_dict: Optional[Dict[str, Any]] = None

    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        env = Environment(dist_env=self.distributed_env, worker_env=worker_env)

        if _should_replace_path(self.input_dir.path):
            # FIXME: Remove the `shard_rank` from the cache_path to enable reloading chunks for the second epoch
            # without paying the cost of re-download
            cache_path = _try_create_cache_dir(
                input_dir=self.input_dir.path if self.input_dir.path else self.input_dir.url, shard_rank=env.shard_rank
            )
            if cache_path is not None:
                self.input_dir.path = cache_path

        cache = Cache(
            input_dir=self.input_dir,
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=self.serializers,
            max_cache_size=self.max_cache_size,
        )
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache

    def _create_shuffler(self, cache: Cache) -> Shuffle:
        seed = self.seed
        drop_last = self.drop_last
        if self._state_dict is not None:
            restart_keys = sorted(self._state_dict)
            state: Dict[str, Any] = self._state_dict[restart_keys[-1]]
            seed = state["seed"]
            drop_last = state["drop_last"]
        return FullShuffle(cache, seed, drop_last) if self.shuffle else NoShuffle(cache, seed, drop_last)

    def __len__(self) -> int:
        if self.shuffler is None:
            cache = self._create_cache(worker_env=_WorkerEnv.detect())
            self.shuffler = self._create_shuffler(cache)
        return self.shuffler.get_len(self.distributed_env, self.current_epoch)

    def __iter__(self) -> "StreamingDataset":
        self.worker_env = _WorkerEnv.detect()
        self.cache = self._create_cache(worker_env=self.worker_env)
        self.shuffler = self._create_shuffler(self.cache)

        # Handle restart
        if self._state_dict:
            self._validate_state_dict()
            restart_keys = sorted(self._state_dict)
            state: Dict[str, Any] = self._state_dict[restart_keys[-1]]
            self.current_epoch = state["current_epoch"]

        chunks_per_replica, intervals_per_replica = self.shuffler.get_chunks_and_intervals_per_ranks(
            self.distributed_env, self.current_epoch
        )
        chunks_replica = chunks_per_replica[self.distributed_env.global_rank % self.distributed_env.world_size]
        intervals_replica = intervals_per_replica[self.distributed_env.global_rank % self.distributed_env.world_size]

        # Handle restart
        if self._state_dict:
            self._resume(chunks_replica, intervals_replica)
        else:
            chunks_per_replica, intervals_per_replica = self.shuffler.get_chunks_and_intervals_per_ranks(
                self.distributed_env, self.current_epoch
            )
            chunks_replica = chunks_per_replica[self.distributed_env.global_rank % self.distributed_env.world_size]
            intervals_replica = intervals_per_replica[
                self.distributed_env.global_rank % self.distributed_env.world_size
            ]

            self.worker_chunks = []
            self.worker_intervals = []

            for i, (chunk_index, chunk_interval) in enumerate(zip(chunks_replica, intervals_replica)):
                if i % self.worker_env.world_size != self.worker_env.rank:
                    continue
                self.worker_chunks.append(chunk_index)
                self.worker_intervals.append(chunk_interval)

            self.num_chunks = len(self.worker_chunks)

            self.current_indexes = []
            self.chunk_index = 0
            self.global_index = 0
            self.index = 0

        self.has_triggered_download = False
        self.last_time = time()

        return self

    def _resume(self, chunks_replica: List[int], intervals_replica: List[Any]) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.shuffler

        restart_keys = sorted(self._state_dict)

        # Get the state from the previous run
        state: Dict[str, Any] = self._state_dict[restart_keys[-1]]

        num_workers = state["num_workers"]
        batch_size = state["batch_size"]

        # TODO: Implement elastic sampling where the number of workers, ranks can change.
        num_samples_yielded = sum([state["num_samples_yielded"] for state in self._state_dict.values()])

        # replay sampling from each worker / chunks using the batch size
        workers_chunks, workers_intervals = _associate_chunks_to_workers(
            num_workers, self.worker_env, chunks_replica, intervals_replica
        )
        indexes = _replay_sampling(num_samples_yielded, batch_size, num_workers)
        chunks_index, indexes = _replay_chunks_sampling(workers_intervals, indexes)

        # select the chunks and intervals associated to this worker
        worker_rank = self.worker_env.rank
        self.num_chunks = len(workers_intervals[worker_rank])
        self.chunk_index = chunks_index[worker_rank]
        self.worker_chunks = workers_chunks[worker_rank]
        self.worker_intervals = workers_intervals[worker_rank]

        # replay the indexes for the current chunks
        interval = workers_intervals[worker_rank][self.chunk_index]
        current_indexes = np.arange(interval[0], interval[1])

        # re-shuffle the indexes
        current_indexes = self.shuffler(current_indexes, self.num_chunks, self.current_epoch, self.chunk_index)

        # skip any indexes already consumed
        current_indexes = current_indexes[indexes[worker_rank] :]
        self.current_indexes = current_indexes

        # bump the chunk_index
        self.chunk_index += 1

    def __getitem__(self, index: Union[ChunkedIndex, int]) -> Any:
        if self.cache is None:
            self.worker_env = _WorkerEnv.detect()
            self.cache = self._create_cache(worker_env=self.worker_env)
            self.shuffler = self._create_shuffler(self.cache)
        if isinstance(index, int):
            index = ChunkedIndex(index, self.cache._get_chunk_index_from_index(index))
        return self.cache[index]

    def __next__(self) -> Any:
        # Prevent to create more batch on a given process
        if self.global_index >= len(self):
            self.current_epoch += 1
            raise StopIteration

        # Lazily re-populate the interval to reduce memory usage.
        if len(self.current_indexes) == 0:
            if self.chunk_index == self.num_chunks:
                self.current_epoch += 1
                raise StopIteration

            # reset index
            self.index = 0

            interval = self.worker_intervals[self.chunk_index]
            current_indexes = np.arange(interval[0], interval[1])

            assert self.shuffler is not None
            assert self.num_chunks is not None
            self.current_indexes = self.shuffler(current_indexes, self.num_chunks, self.current_epoch, self.chunk_index)

            self.chunk_index += 1

        # Get the first index
        index = self.current_indexes.pop(0)

        # Call the `__getitem__` method.
        data = self.__getitem__(
            ChunkedIndex(
                index=index,
                chunk_index=self.worker_chunks[self.chunk_index - 1],
                # We provide the chunks indexes only one the first
                chunk_indexes=None if self.has_triggered_download else self.worker_chunks,
                is_last_index=(self.chunk_index - 1) == len(self.worker_intervals) and len(self.current_indexes) == 1,
            )
        )

        self.has_triggered_download = True
        self.global_index += 1
        self.index += 1

        return data

    def state_dict(self, num_samples_yielded: int, num_workers: int, batch_size: int) -> Dict[str, Any]:
        if _is_in_dataloader_worker():
            raise RuntimeError("The method `state_dict` should only be called in the main process.")

        state = {
            "num_samples_yielded": num_samples_yielded,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "current_epoch": self.current_epoch,
            "input_dir_path": self.input_dir.path,
            "input_dir_url": self.input_dir.url,
            "item_loader": self.item_loader.state_dict() if self.item_loader else None,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "world_size": self.distributed_env.world_size,
            "shuffle": self.shuffle,
        }

        if self._state_dict:
            num_restarts = len(self._state_dict)
            return {**self._state_dict, f"{num_restarts}": state}
        return {"0": state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict:
            # the state is restored within the workers
            self._state_dict = state_dict

    def _validate_state_dict(self) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.cache

        restart_keys = sorted(self._state_dict)
        state: Dict[str, Any] = self._state_dict[restart_keys[-1]]

        if state["shuffle"] != self.shuffle:
            raise ValueError(
                "The provided `shuffle` state doesn't match the current one. "
                f"Found `{self.shuffle}` instead of `{state['shuffle']}`."
            )

        if state["num_workers"] != self.worker_env.world_size:
            raise ValueError(
                "The provided `num_workers` state doesn't match the current one. "
                f"Found `{self.worker_env.world_size}` instead of `{state['num_workers']}`."
            )

        if state["input_dir_path"] != self.input_dir.path:
            raise ValueError(
                "The provided `input_dir` path state doesn't match the current one. "
                f"Found `{self.input_dir.path}` instead of `{state['input_dir_path']}`."
            )

        if state["input_dir_url"] != self.input_dir.url:
            raise ValueError(
                "The provided `input_dir` URL state doesn't match the current one. "
                f"Found `{self.input_dir.url}` instead of `{state['input_dir_url']}`."
            )

        if state["seed"] != self.seed:
            raise ValueError(
                "The provided `seed` state doesn't match the current one. "
                f"Found `{self.seed}` instead of `{state['seed']}`."
            )

        if self.item_loader and state["item_loader"] != self.item_loader.state_dict():
            raise ValueError(
                "The provided `item_loader` state doesn't match the current one. "
                f"Found `{self.item_loader.state_dict()}` instead of `{state['item_loader']}`."
            )

        if state["drop_last"] != self.drop_last:
            raise ValueError(
                "The provided `drop_last` state doesn't match the current one. "
                f"Found `{self.drop_last}` instead of `{state['drop_last']}`."
            )


def _try_create_cache_dir(input_dir: Optional[str], shard_rank: int = 0) -> Optional[str]:
    hash_object = hashlib.md5((input_dir or "").encode())
    if "LIGHTNING_CLUSTER_ID" not in os.environ or "LIGHTNING_CLOUD_PROJECT_ID" not in os.environ:
        cache_dir = os.path.join(_DEFAULT_CACHE_DIR, hash_object.hexdigest(), str(shard_rank))
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    cache_dir = os.path.join("/cache", "chunks", hash_object.hexdigest(), str(shard_rank))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _should_replace_path(path: Optional[str]) -> bool:
    """Whether the input path is a special path to be replaced."""
    if path is None or path == "":
        return True

    return "/datasets/" in path or "_connections/" in path


def _is_in_dataloader_worker() -> bool:
    return get_worker_info() is not None


def is_integer(value: str) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def _associate_chunks_to_workers(
    num_workers: int, worker_env: _WorkerEnv, chunks_replica: List[int], intervals_replica: List[Any]
) -> Any:
    workers_chunks = {}
    workers_intervals = {}

    for worker_idx in range(num_workers):
        worker_chunks = []
        worker_intervals = []
        for i, (chunk_index, chunk_interval) in enumerate(zip(chunks_replica, intervals_replica)):
            if i % worker_env.world_size != worker_idx:
                continue

            worker_chunks.append(chunk_index)
            worker_intervals.append(chunk_interval)

        workers_chunks[worker_idx] = worker_chunks
        workers_intervals[worker_idx] = worker_intervals

    return workers_chunks, workers_intervals


def _replay_sampling(num_samples_yielded: int, batch_size: int, num_workers: int) -> Dict[int, int]:
    """This function replays the sampling from the dataloader."""
    divisible_num_batches_yielded = num_samples_yielded // (num_workers * batch_size)

    indexes = {}
    for worker_idx in range(num_workers):
        indexes[worker_idx] = divisible_num_batches_yielded * batch_size

    num_samples_yielded = num_samples_yielded - (num_workers * divisible_num_batches_yielded * batch_size)

    # take care of the reminder
    worker_idx = 0  # reset the worker_idx
    while True:
        if num_samples_yielded >= batch_size:
            indexes[worker_idx] += batch_size
            worker_idx = (worker_idx + 1) % num_workers
            num_samples_yielded -= batch_size
        else:
            indexes[worker_idx] += num_samples_yielded
            break
    return indexes


def _replay_chunks_sampling(
    workers_intervals: Dict[int, List[Any]], indexes: Dict[int, int]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    chunks_index = {}

    for worker_idx in range(len(workers_intervals)):
        chunks_index[worker_idx] = 0

    for worker_idx, intervals in workers_intervals.items():
        for interval in intervals:
            size = interval[-1] - interval[0]
            if indexes[worker_idx] >= size:
                indexes[worker_idx] -= size
                chunks_index[worker_idx] += 1

    return chunks_index, indexes
