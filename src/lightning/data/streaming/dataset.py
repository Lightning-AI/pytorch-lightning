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

import copy
import hashlib
import os
from typing import Any, List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset

from lightning.data.datasets.env import Environment, _DistributedEnv, _WorkerEnv
from lightning.data.streaming import Cache
from lightning.data.streaming.constants import _INDEX_FILENAME, _LIGHTNING_CLOUD_LATEST
from lightning.data.streaming.item_loader import BaseItemLoader
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.shuffle import FullShuffle, NoShuffle, Shuffle

if _LIGHTNING_CLOUD_LATEST:
    from lightning_cloud.resolver import _resolve_dir


class StreamingDataset(IterableDataset):
    """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class."""

    def __init__(
        self,
        input_dir: str,
        item_loader: Optional[BaseItemLoader] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class.

        Arguments:
            input_dir: Path to the folder where the input data is stored.
            item_loader: The logic to load an item from a chunk.
            shuffle: Whether to shuffle the data.
            drop_last: If `True`, drops the last items to ensure that
                all processes/workers return the same amount of data.
            seed: Random seed for shuffling.

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

        self.cache: Optional[Cache] = None
        self.distributed_env = _DistributedEnv.detect()
        self.worker_env: Optional[_WorkerEnv] = None
        self.worker_chunks: List[int] = []
        self.worker_intervals: List[List[int]] = []
        self.current_indexes: List[int] = []
        self.chunk_index = 0
        self.index = 0
        self.has_triggered_download = False
        self.min_items_per_replica: Optional[int] = None
        self.current_epoch = 0
        self.random_state = None
        self.shuffler: Optional[Shuffle] = None

    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        env = Environment(dist_env=self.distributed_env, worker_env=worker_env)
        cache_path = _try_create_cache_dir(input_dir=self.input_dir.path, shard_rank=env.shard_rank)
        cache_dir = copy.deepcopy(self.input_dir)
        if cache_path:
            cache_dir.path = cache_path

        cache = Cache(input_dir=cache_dir, item_loader=self.item_loader, chunk_bytes=1)
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache

    def _create_shuffler(self, cache: Cache) -> Shuffle:
        return (
            FullShuffle(cache, self.seed, self.drop_last)
            if self.shuffle
            else NoShuffle(cache, self.seed, self.drop_last)
        )

    def __len__(self) -> int:
        if self.shuffler is None:
            cache = self._create_cache(worker_env=_WorkerEnv.detect())
            self.shuffler = self._create_shuffler(cache)
        return self.shuffler.get_len(self.distributed_env, self.current_epoch)

    def __iter__(self) -> "StreamingDataset":
        self.worker_env = _WorkerEnv.detect()
        self.cache = self._create_cache(worker_env=self.worker_env)
        self.shuffler = self._create_shuffler(self.cache)

        chunks_per_replica, intervals_per_replica = self.shuffler.get_chunks_and_intervals_per_ranks(
            self.distributed_env, self.current_epoch
        )
        current_chunks = chunks_per_replica[self.distributed_env.global_rank % self.distributed_env.world_size]
        current_intervals = intervals_per_replica[self.distributed_env.global_rank % self.distributed_env.world_size]

        self.worker_chunks = []
        self.worker_intervals = []

        for i, (chunk_index, chunk_interval) in enumerate(zip(current_chunks, current_intervals)):
            if i % self.worker_env.world_size != self.worker_env.rank:
                continue
            self.worker_chunks.append(chunk_index)
            self.worker_intervals.append(chunk_interval)

        self.current_indexes = []
        self.chunk_index = 0
        self.index = 0

        return self

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
        if self.index >= len(self):
            self.current_epoch += 1
            raise StopIteration

        # Lazily re-populate the interval to reduce memory usage.
        if len(self.current_indexes) == 0:
            if self.chunk_index == len(self.worker_intervals):
                self.current_epoch += 1
                raise StopIteration

            interval = self.worker_intervals[self.chunk_index]
            current_indexes = np.arange(interval[0], interval[1])

            assert self.shuffler is not None
            self.current_indexes = self.shuffler(current_indexes)
            self.chunk_index += 1

        # Get the first index
        index = self.current_indexes.pop(0)

        # Call the `__getitem__` method.
        data = self.__getitem__(
            ChunkedIndex(
                index=index,
                chunk_index=self.worker_chunks[self.chunk_index - 1],
                chunk_indexes=None if self.has_triggered_download else self.worker_chunks,
            )
        )

        self.has_triggered_download = True
        self.index += 1

        return data


def _try_create_cache_dir(input_dir: str, shard_rank: int = 0) -> Optional[str]:
    if "LIGHTNING_CLUSTER_ID" not in os.environ or "LIGHTNING_CLOUD_PROJECT_ID" not in os.environ:
        return None
    hash_object = hashlib.md5(input_dir.encode())
    cache_dir = os.path.join("/cache", "chunks", hash_object.hexdigest(), str(shard_rank))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
