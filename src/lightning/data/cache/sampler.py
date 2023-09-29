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

import logging
from typing import Iterator, List

import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler, Sized

from lightning.data.cache import Cache

logger = logging.Logger(__name__)


class IteratorSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from

    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)


class CacheSampler(Sampler):
    def __init__(self, dataset_size: int, num_workers: int, batch_size: int):
        """The CacheSampler splits the dataset indices into ordered chunks and assign each one of them to a DataLoader
        worker. The Cache Writer expects the index to be provided in an ordered fashion.

        Arguments:
            dataset_size: The size of the dataset.
            num_workers: The number of workers provided to the DataLoader
            batch_size: The number of items in a batch

        """

        super().__init__(None)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.indices = range(dataset_size)
        self.dataset_size = dataset_size
        worker_size = dataset_size // self.num_workers
        self.samplers = []
        for worker_idx in range(num_workers):
            is_last = worker_idx == num_workers - 1
            worker_indices = self.indices[
                worker_idx * worker_size : dataset_size if is_last else (worker_idx + 1) * worker_size
            ]
            self.samplers.append(IteratorSampler(worker_indices))
        self.iterators = []
        self._done = set()
        assert sum([len(s) for s in self.samplers]) == dataset_size
        self.worker_id = 0
        self.indice_id = 0

    def __len__(self) -> int:
        return self.dataset_size

    @property
    def done(self) -> bool:
        return len(self._done) == len(self.iterators)

    def __iter__(self) -> "CacheSampler":
        self._done = set()

        for sampler in self.samplers:
            self.iterators.append(iter(sampler))

        return self

    def __next__(self) -> List[int]:
        while len(self._done) != self.iterators:
            try:
                data = next(self.iterators[self.worker_id])
                self.indice_id += 1
                if self.indice_id == self.batch_size:
                    self.indice_id = 0
                    self.worker_id = (self.worker_id + 1) % self.num_workers
                return data
            except StopIteration:
                self._done.add(self.worker_id)
                self.indice_id = 0
                self.worker_id = (self.worker_id + 1) % self.num_workers
                raise StopIteration


class DistributedCacheSampler(Sampler):
    def __init__(self, dataset_size: int, num_replicas: int, rank: int, num_workers: int, batch_size: int):
        """The DistributedCacheSampler splits the dataset indices into ordered chunks along all the replicas and their
        workers. The Cache Writer expects the index to be provided in an ordered fashion.

        Arguments:
            dataset_size: The size of the dataset.
            num_workers: The number of workers provided to the DataLoader
            batch_size: The number of items in a batch

        """
        super().__init__(None)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.indices = range(dataset_size)
        self.dataset_size = dataset_size
        replica_size = dataset_size // num_replicas
        worker_size = dataset_size // (num_replicas * self.num_workers)
        self.samplers = []
        for replica_idx in range(num_replicas):
            if replica_idx != rank:
                continue

            is_last_replica = replica_idx == num_replicas - 1
            start_replica = replica_idx * replica_size
            end_replica = dataset_size if is_last_replica else (replica_idx + 1) * replica_size
            replica_indices = self.indices[start_replica:end_replica]

            replica_size = len(replica_indices)

            for worker_idx in range(num_workers):
                is_last_worker = worker_idx == num_workers - 1
                start_worker = worker_idx * worker_size
                end_worker = replica_size if is_last_worker else (worker_idx + 1) * worker_size
                worker_indices = replica_indices[start_worker:end_worker]
                self.samplers.append(IteratorSampler(worker_indices))

        self.iterators = []
        self._done = set()

        assert sum([len(s) for s in self.samplers]) == replica_size
        self.worker_id = 0
        self.indice_id = 0

    def __len__(self) -> int:
        return self.dataset_size

    @property
    def done(self) -> bool:
        return len(self._done) == len(self.iterators)

    def __iter__(self) -> "DistributedCacheSampler":
        self._done = set()

        for sampler in self.samplers:
            self.iterators.append(iter(sampler))

        return self

    def __next__(self) -> List[str]:
        while len(self._done) != self.iterators:
            try:
                data = next(self.iterators[self.worker_id])
                self.indice_id += 1
                if self.indice_id == self.batch_size:
                    self.indice_id = 0
                    self.worker_id = (self.worker_id + 1) % self.num_workers
                return data
            except StopIteration:
                self._done.add(self.worker_id)
                self.indice_id = 0
                self.worker_id = (self.worker_id + 1) % self.num_workers
                raise StopIteration


class CacheBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset_size: int,
        num_replicas: int,
        rank: int,
        num_workers: int,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        cache: Cache,
    ):
        """The CacheBatchSampler handles the generation of batch indices.

        If the cache isn't filled, the batch sampler alternates with ordered indices for the writer to chunk the dataset
        If the cache is filled, it acts as normal BatchSampler.

        Arguments:
            dataset_size: The size of the dataset.
            num_replicas: The number of processes involves in the distributed training.
            rank: The rank of the given process
            num_workers: The number of workers provided to the DataLoader.
            batch_size: The number of items in a batch.
            shuffle: Whether the data should be shuffled.
            cache: The cache associated to the dataset.

        """

        if num_replicas == 1:
            if not cache.filled and num_workers > 1:
                sampler = CacheSampler(dataset_size, num_workers, batch_size)
            elif shuffle:
                sampler = RandomSampler(range(dataset_size))
            else:
                sampler = SequentialSampler(range(dataset_size))
        else:
            if not cache.filled:
                sampler = DistributedCacheSampler(dataset_size, num_replicas, rank, num_workers, batch_size)
            else:
                sampler = DistributedSampler(range(dataset_size), num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        super().__init__(sampler, batch_size, drop_last)
        self._num_replicas = num_replicas
        self._rank = rank
        self._cache = cache
        self._shuffle = shuffle
        self._num_workers = num_workers

    def __iter_ordered__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        iterator = iter(self.sampler)
        batch = []
        while not self.sampler.done:
            try:
                idx = next(iterator)
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            except StopIteration:
                if self.sampler.done:
                    yield batch
                    return
                yield batch
                batch = []

    def __iter__(self):
        if self._cache.filled and self._shuffle:
            return self.__iter_from_chunks__()
        if self._num_workers > 1 and not self._cache.filled:
            return self.__iter_ordered__()
        return super().__iter__()

    def __iter_from_chunks__(self):
        chunk_intervals = self._cache.get_chunk_interval()
        shuffled_chunk_intervals = np.random.permutation(chunk_intervals)

        if self._num_replicas == 1:
            indices = []
            for interval in shuffled_chunk_intervals:
                interval_indices = np.arange(interval[0], interval[1])
                shuffled_interval_indices = np.random.permutation(interval_indices)
                indices.extend(shuffled_interval_indices.tolist())

            if len(indices) != len(self.sampler):
                raise Exception("The generated indices don't match the initial length of the sampler.")

        else:
            chunks_per_replica = len(shuffled_chunk_intervals) // self._num_replicas
            for replica_idx in range(self._num_replicas):
                if replica_idx != self._rank:
                    continue
                is_last_replica = replica_idx == self._num_replicas - 1
                start_replica = replica_idx * chunks_per_replica
                end_replica = len(chunk_intervals) if is_last_replica else (replica_idx + 1) * chunks_per_replica
                shuffled_chunk_intervals_replica = shuffled_chunk_intervals[start_replica:end_replica]

                indices = []
                for interval in shuffled_chunk_intervals_replica:
                    interval_indices = np.arange(interval[0], interval[1])
                    shuffled_interval_indices = np.random.permutation(interval_indices)
                    indices.extend(shuffled_interval_indices.tolist())

        self.sampler = IteratorSampler(indices)

        return super().__iter__()

    def __len__(self) -> int:
        return super().__len__()
