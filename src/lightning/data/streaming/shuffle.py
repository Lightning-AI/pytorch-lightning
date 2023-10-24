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
    def permute(self, array: np.ndarray) -> List[int]:
        pass


class NoShuffle(Shuffle):
    @lru_cache
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_process = self.get_chunks_and_intervals_per_process(distributed_env, current_epoch)
        min_items_per_process = min(
            [sum([(interval[-1] - interval[0]) for interval in intervals]) for intervals in intervals_per_process]
        )
        return min_items_per_process

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

    def permute(self, array: np.ndarray) -> List[int]:
        return array.tolist()


class MinShuffle(Shuffle):
    @lru_cache
    def get_len(self, distributed_env: _DistributedEnv, current_epoch: int) -> int:
        _, intervals_per_process = self.get_chunks_and_intervals_per_process(distributed_env, current_epoch)
        min_items_per_process = min(
            [sum([(interval[-1] - interval[0]) for interval in intervals]) for intervals in intervals_per_process]
        )
        return min_items_per_process

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

    def permute(self, array: np.ndarray) -> List[int]:
        assert self.random_state
        return self.random_state.permutation(array).tolist()
