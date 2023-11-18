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
from typing import Any, List, Tuple

import numpy as np

from lightning.data.streaming import Cache
from lightning.data.utilities.env import _DistributedEnv


class Shuffle(ABC):
    """Shuffle describe how to distribute chunked datasets across processes and workers."""

    def __init__(self, cache: Cache, seed: int, drop_last: bool):
        self.cache = cache
        self.seed = seed
        self.drop_last = drop_last
        self.random_state = None

    @lru_cache(maxsize=10)
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_ranks = self.get_chunks_and_intervals_per_ranks(distributed_env, current_epoch)

        if self.drop_last:
            items_per_process = [
                sum((interval[-1] - interval[0]) for interval in intervals) for intervals in intervals_per_ranks
            ]
            min_items_per_process = min(items_per_process)
            return min_items_per_process

        return sum((interval[-1] - interval[0]) for interval in intervals_per_ranks[distributed_env.global_rank])

    @abstractmethod
    def get_chunks_and_intervals_per_ranks(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        pass

    @abstractmethod
    def __call__(self, array: np.ndarray) -> List[int]:
        pass


class NoShuffle(Shuffle):
    """NoShuffle doesn't shuffle the items and ensure all the processes receive the same number of items if drop_last
    is True."""

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_ranks(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        chunk_intervals = self.cache.get_chunk_intervals()
        chunks_per_ranks: List[List[int]] = [[] for _ in range(distributed_env.world_size)]
        intervals_per_ranks: List[List[Tuple]] = [[] for _ in range(distributed_env.world_size)]
        for chunk_index, chunk_interval in enumerate(chunk_intervals):
            replica_index = chunk_index % distributed_env.world_size
            chunks_per_ranks[replica_index].append(chunk_index)
            intervals_per_ranks[replica_index].append(chunk_interval)

        return chunks_per_ranks, intervals_per_ranks

    def __call__(self, array: np.ndarray) -> List[int]:
        return array.tolist()


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
    def get_chunks_and_intervals_per_ranks(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        self.random_state = np.random.RandomState(seed=self.seed + current_epoch)  # type: ignore

        # 1. Get the intervals
        chunk_intervals = self.cache.get_chunk_intervals()

        # 2. Shuffle them
        indexes = range(len(chunk_intervals))
        shuffled_indexes = self.random_state.permutation(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes]

        # 3. Compute the items budget of each rank
        num_items = sum([(interval[-1] - interval[0]) for interval in chunk_intervals])
        num_items_per_ranks: List[int] = [
            num_items // distributed_env.world_size + num_items % distributed_env.world_size
            if rank == distributed_env.world_size - 1 and not self.drop_last
            else num_items // distributed_env.world_size
            for rank in range(distributed_env.world_size)
        ]
        chunks_per_ranks: List[List[int]] = [[] for _ in range(distributed_env.world_size)]
        intervals_per_ranks: List[List[List[int]]] = [[] for _ in range(distributed_env.world_size)]

        # 4. Assign the chunk & intervals to each rank
        for chunk_index, chunk_interval in zip(shuffled_indexes, shuffled_chunk_intervals):
            rank = 0

            while True:
                if rank == len(num_items_per_ranks):
                    break

                items_left_to_assign = num_items_per_ranks[rank]

                if items_left_to_assign == 0:
                    rank += 1
                    continue

                items_in_chunk = chunk_interval[-1] - chunk_interval[0]

                if items_in_chunk == 0:
                    break

                if items_in_chunk > items_left_to_assign:
                    chunks_per_ranks[rank].append(chunk_index)
                    begin, end = chunk_interval
                    intervals_per_ranks[rank].append([begin, begin + items_left_to_assign])
                    chunk_interval = (begin + items_left_to_assign, end)
                    num_items_per_ranks[rank] = 0
                    rank += 1
                else:
                    chunks_per_ranks[rank].append(chunk_index)
                    intervals_per_ranks[rank].append(chunk_interval)
                    num_items_per_ranks[rank] -= items_in_chunk
                    break

        return chunks_per_ranks, intervals_per_ranks

    def __call__(self, array: np.ndarray) -> List[int]:
        assert self.random_state
        return self.random_state.permutation(array).tolist()
