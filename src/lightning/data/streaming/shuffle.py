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

from lightning.data.streaming import Cache
from lightning.data.utilities.env import _DistributedEnv
from lightning.data.utilities.shuffle import _associate_chunks_and_internals_to_ranks, _intra_node_chunk_shuffle


class Shuffle(ABC):
    """Shuffle describe how to distribute chunked datasets across processes and workers."""

    def __init__(self, cache: Cache, seed: int, drop_last: bool):
        self.cache = cache
        self.seed = seed
        self.drop_last = drop_last

    @lru_cache(maxsize=10)
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_ranks = self.get_chunks_and_intervals_per_ranks(distributed_env, current_epoch)

        if self.drop_last:
            items_per_process = [
                sum((interval[-1] - interval[0]) for interval in intervals) for intervals in intervals_per_ranks
            ]
            # Validate each processes gets the exact number of elements
            if len(items_per_process) > 1:
                assert all(items_per_process[0] == items_to_process for items_to_process in items_per_process[:1])
            return items_per_process[0]

        return sum((interval[-1] - interval[0]) for interval in intervals_per_ranks[distributed_env.global_rank])

    @abstractmethod
    def get_chunks_and_intervals_per_ranks(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        pass

    @abstractmethod
    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> List[int]:
        pass


class NoShuffle(Shuffle):
    """NoShuffle doesn't shuffle the items and ensure all the processes receive the same number of items if drop_last
    is True."""

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_ranks(self, distributed_env: _DistributedEnv, current_epoch: int) -> Any:
        # 1. Get the intervals
        chunk_intervals = self.cache.get_chunk_intervals()
        indexes = range(len(chunk_intervals))

        # 2. Compute the items budget of each rank
        chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
            distributed_env, indexes, chunk_intervals, self.drop_last
        )

        return chunks_per_ranks, intervals_per_ranks

    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> List[int]:
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
        # 1. Get the intervals
        chunk_intervals = self.cache.get_chunk_intervals()

        # 2. Shuffle them
        indexes = range(len(chunk_intervals))

        # If we have multiple nodes, the seed_shift is constant here.
        # Here is why. When you are running epoch 1, we need to shuffle the chunks
        # and associate to each rank. This is done there.
        # When you are running epoch 2 or more, we need to keep the same shuffling
        # than in epoch 1 because shuffle a second time within the node.
        # This is done slighyly down this function.
        seed_shift = 1 if distributed_env.num_nodes > 1 else current_epoch
        shuffled_indexes = np.random.RandomState(seed=self.seed + seed_shift).permutation(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes].tolist()

        # 3. Compute the items budget of each rank
        chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
            distributed_env, shuffled_indexes, shuffled_chunk_intervals, self.drop_last
        )

        # For the first epoch, no need of further shuffling
        if current_epoch == 1 or distributed_env.num_nodes == 1:
            return chunks_per_ranks, intervals_per_ranks

        # Perform shuffle within the nodes to avoid cache miss.
        # Note: It is possible for the overlapping chunks to change due to the changing order.
        shuffled_indexes = _intra_node_chunk_shuffle(distributed_env, chunks_per_ranks, self.seed, current_epoch)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes].tolist()

        chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
            distributed_env, shuffled_indexes, shuffled_chunk_intervals, self.drop_last
        )

        return chunks_per_ranks, intervals_per_ranks

    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> List[int]:
        return np.random.RandomState([self.seed, num_chunks * current_epoch, chunk_index]).permutation(array).tolist()
