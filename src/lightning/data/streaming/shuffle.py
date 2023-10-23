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
from typing import Any, List

import numpy as np

from lightning.data.datasets.env import _DistributedEnv
from lightning.data.streaming import Cache


class Shuffle(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_chunks_and_intervals_per_replica(self) -> Any:
        pass


class NoShuffle(Shuffle):
    def __init__(self, cache: Cache, seed: int, distributed_env: _DistributedEnv, num_iter: int):
        self.cache = cache
        self.seed = seed
        self.distributed_env = distributed_env
        self.num_iter = num_iter

    def __len__(self) -> int:
        _, intervals_per_replica = self.get_chunks_and_intervals_per_replica()
        min_items_per_replica = min(
            [sum([(interval[-1] - interval[0]) for interval in intervals]) for intervals in intervals_per_replica]
        )
        return min_items_per_replica

    def get_chunks_and_intervals_per_replica(self) -> Any:
        self.random_state = np.random.RandomState(seed=self.seed + self.num_iter)  # type: ignore
        chunk_intervals = self.cache.get_chunk_interval()
        indexes = list(range(len(chunk_intervals)))
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[indexes]

        chunks_per_replica: List[List[int]] = [[] for _ in range(self.distributed_env.world_size)]
        intervals_per_replica: List[List[List[int]]] = [[] for _ in range(self.distributed_env.world_size)]
        for index, (chunk_index, chunk_interval) in enumerate(zip(indexes, shuffled_chunk_intervals)):
            replica_index = index % self.distributed_env.world_size
            chunks_per_replica[replica_index].append(chunk_index)
            intervals_per_replica[replica_index].append(chunk_interval)

        return chunks_per_replica, intervals_per_replica


class MinShuffle(Shuffle):
    def __init__(self, cache: Cache, seed: int, distributed_env: _DistributedEnv, num_iter: int):
        self.cache = cache
        self.seed = seed
        self.distributed_env = distributed_env
        self.num_iter = num_iter

    def __len__(self) -> int:
        _, intervals_per_replica = self.get_chunks_and_intervals_per_replica()
        min_items_per_replica = min(
            [sum([(interval[-1] - interval[0]) for interval in intervals]) for intervals in intervals_per_replica]
        )
        return min_items_per_replica

    def get_chunks_and_intervals_per_replica(self) -> Any:
        self.random_state = np.random.RandomState(seed=self.seed + num_iter)  # type: ignore
        chunk_intervals = self.cache.get_chunk_interval()
        indexes = range(len(chunk_intervals))
        shuffled_indexes = self.random_state.permutation(indexes) if self.shuffle else list(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes]

        chunks_per_replica: List[List[int]] = [[] for _ in range(self.distributed_env.world_size)]
        intervals_per_replica: List[List[List[int]]] = [[] for _ in range(self.distributed_env.world_size)]
        for index, (chunk_index, chunk_interval) in enumerate(zip(shuffled_indexes, shuffled_chunk_intervals)):
            replica_index = index % self.distributed_env.world_size
            chunks_per_replica[replica_index].append(chunk_index)
            intervals_per_replica[replica_index].append(chunk_interval)

        return chunks_per_replica, intervals_per_replica
