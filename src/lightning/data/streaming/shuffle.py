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

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, List

import numpy as np

from lightning.data.datasets.env import _DistributedEnv
from lightning.data.streaming import Cache


class Shuffle(ABC):
    """Shuffle describe how to distribute chunked datasets across processes and workers."""

    def __init__(self, cache: Cache, seed: int):
        self.cache = cache
        self.seed = seed
        self.random_state = None

    @abstractmethod
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        pass

    @abstractmethod
    def get_chunks_and_intervals_per_process(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        pass

    @abstractmethod
    def __call__(self, array: np.ndarray) -> List[int]:
        pass


class NoShuffle(Shuffle):
    """NoShuffle doesn't shuffle the items and ensure all the processes receive the same number of items."""

    @lru_cache(maxsize=10)
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_process = self.get_chunks_and_intervals_per_process(distributed_env, current_epoch)
        min_items_per_process = min(
            [sum([(interval[-1] - interval[0]) for interval in intervals]) for intervals in intervals_per_process]
        )
        return min_items_per_process

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_process(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        self.random_state = np.random.RandomState(seed=self.seed + current_epoch)  # type: ignore
        chunk_intervals = self.cache.get_chunk_intervals()
        indexes = list(range(len(chunk_intervals)))
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[indexes]

        chunks_per_process: List[List[int]] = [[] for _ in range(distributed_env.world_size)]
        intervals_per_process: List[List[List[int]]] = [[] for _ in range(distributed_env.world_size)]
        for index, (chunk_index, chunk_interval) in enumerate(zip(indexes, shuffled_chunk_intervals)):
            replica_index = index % distributed_env.world_size
            chunks_per_process[replica_index].append(chunk_index)
            intervals_per_process[replica_index].append(chunk_interval)

        return chunks_per_process, intervals_per_process

    def __call__(self, array: np.ndarray) -> List[int]:
        return array.tolist()


class TruncatedShuffle(Shuffle):
    """TruncatedShuffle shuffles the chunks and associates them to the ranks.

    As the number of items in a chunk varies, it is possible for a rank to end up with more or less items.

    To ensure the same fixed dataset length for all ranks, we compute the minimum number of items across all ranks.

    For the ranks with more items than the minimum, the remaining items are dropped.

    Note: This is the fastest sampling strategy but at the cost of losing items.

    """

    @lru_cache(maxsize=10)
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_process = self.get_chunks_and_intervals_per_process(distributed_env, current_epoch)
        min_items_per_process = min(
            [sum([(interval[-1] - interval[0]) for interval in intervals]) for intervals in intervals_per_process]
        )
        return min_items_per_process

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_process(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        self.random_state = np.random.RandomState(seed=self.seed + current_epoch)  # type: ignore
        chunk_intervals = self.cache.get_chunk_intervals()
        indexes = range(len(chunk_intervals))
        shuffled_indexes = self.random_state.permutation(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes]

        chunks_per_process: List[List[int]] = [[] for _ in range(distributed_env.world_size)]
        intervals_per_process: List[List[List[int]]] = [[] for _ in range(distributed_env.world_size)]
        for index, (chunk_index, chunk_interval) in enumerate(zip(shuffled_indexes, shuffled_chunk_intervals)):
            replica_index = index % distributed_env.world_size
            chunks_per_process[replica_index].append(chunk_index)
            intervals_per_process[replica_index].append(chunk_interval)

        return chunks_per_process, intervals_per_process

    def __call__(self, array: np.ndarray) -> List[int]:
        assert self.random_state
        return self.random_state.permutation(array).tolist()


class FullShuffle(Shuffle):
    """FullShuffle shuffles the chunks and associates them to the ranks.

    As the number of items in a chunk varies, it is possible for a rank to end up with more or less items.

    To ensure the same fixed dataset length for all ranks while dropping as few items as possible,

    we adopt the following strategy.

    We compute the maximum number of items per rank (M) and iterate through the chunks and ranks

    until we have associated at least M items per rank.

    As a result, we lose at most (number of ranks) items. However, as some chunks are shared across ranks. This leads to
    the same chunk to be downloaded multiple times.

    """

    @lru_cache(maxsize=10)
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_process = self.get_chunks_and_intervals_per_process(distributed_env, current_epoch)
        min_items_per_process = min([sum([(i[-1] - i[0]) for i in intervals]) for intervals in intervals_per_process])
        return min_items_per_process

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_process(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        self.random_state = np.random.RandomState(seed=self.seed + current_epoch)  # type: ignore
        chunk_intervals = self.cache.get_chunk_intervals()
        indexes = range(len(chunk_intervals))
        shuffled_indexes = self.random_state.permutation(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes]

        num_items = sum([(interval[-1] - interval[0]) for interval in chunk_intervals])
        num_items_per_process: List[int] = [
            num_items // distributed_env.world_size for _ in range(distributed_env.world_size)
        ]
        chunks_per_process: List[List[int]] = [[] for _ in range(distributed_env.world_size)]
        intervals_per_process: List[List[List[int]]] = [[] for _ in range(distributed_env.world_size)]
        for chunk_index, chunk_interval in zip(shuffled_indexes, shuffled_chunk_intervals):
            process_index = 0

            while True:
                if process_index == len(num_items_per_process):
                    break

                items_left_to_assign = num_items_per_process[process_index]

                if items_left_to_assign == 0:
                    process_index += 1
                    continue

                items_in_chunk = chunk_interval[-1] - chunk_interval[0]

                if items_in_chunk == 0:
                    break

                if items_in_chunk > items_left_to_assign:
                    chunks_per_process[process_index].append(chunk_index)
                    begin, end = chunk_interval
                    intervals_per_process[process_index].append([begin, begin + items_left_to_assign])
                    chunk_interval = (begin + items_left_to_assign + 1, end)
                    num_items_per_process[process_index] = 0
                    process_index += 1
                else:
                    chunks_per_process[process_index].append(chunk_index)
                    intervals_per_process[process_index].append(chunk_interval)
                    num_items_per_process[process_index] -= items_in_chunk
                    break

        return chunks_per_process, intervals_per_process

    def __call__(self, array: np.ndarray) -> List[int]:
        assert self.random_state
        return self.random_state.permutation(array).tolist()
