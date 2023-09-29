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
import os
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler, Sized

from lightning.data.cache.reader import BinaryReader
from lightning.data.cache.writer import BinaryWriter
from lightning.data.datasets.env import _DistributedEnv

logger = logging.Logger(__name__)


class Cache:
    def __init__(
        self,
        cache_dir: str,
        data_format: Union[Dict[str, any], str],
        compression: Optional[str] = None,
        chunk_size: int = 2 << 26,
    ):
        """The Cache enables to optimise dataset format for cloud training. This is done by grouping several elements
        together in order to accelerate fetching.

        Arguments:
            cache_dir: The path to where the chunks will be stored.
            data_format: The structure of the data to be serialized.
            compression: The name of the algorithm to reduce the size of the chunks
            chunk_size: The maximum byte size of chunk.

        """
        super().__init__()
        self._writer = BinaryWriter(cache_dir, data_format, chunk_size=chunk_size, compression=compression)
        self._reader = BinaryReader(cache_dir, compression=compression)
        self._cache_dir = cache_dir
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()
        self._num_workers: Optional[int] = None

    def _setup(self, num_workers: int) -> None:
        self._num_workers = num_workers

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._num_workers is None:
            raise Exception("The Cache wasn't setup properly. HINT: Did you use the CacheDataLoader ?")
        if self._is_done:
            return True
        files = os.listdir(self._cache_dir)
        index_files = [f for f in files if f.endswith("index.json")]
        self._is_done = len(index_files) == self._distributed_env.world_size * (self._num_workers or 1)
        return self._is_done

    def __setitem__(self, index, data) -> None:
        """Store an item in the writer."""
        self._writer[index] = data

    def __getitem__(self, index) -> Dict[str, Any]:
        """Read an item in the reader."""
        return self._reader.read(index)

    def done(self) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer.done()

    def __len__(self) -> int:
        return self._reader.get_length()

    def get_chunk_interval(self):
        return self._reader.get_chunk_interval()


class _SingleProcessDataLoaderIterPatch(_SingleProcessDataLoaderIter):
    """This is overriden to inform the cache is done chunking."""

    def _next_data(self):
        try:
            return super()._next_data()
        except StopIteration:
            for v in self._dataset_fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()
            raise StopIteration()


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


class CacheCollateFn:
    def __init__(self):
        self.collate_fn = default_collate

    def __call__(self, items):
        if all(item is None for item in items):
            return None
        return self.collate_fn(items)


class _MultiProcessingDataLoaderIterPatch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        # Patch PyTorch worker loop
        from torch.utils.data._utils import worker

        from lightning.data.cache.worker import _worker_loop

        worker._worker_loop = _worker_loop
        super().__init__(loader)


class CacheDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        *args,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        shuffle: bool = False,
        generator=None,
        batch_size=1,
        drop_last=False,
        **kwargs,
    ):
        if sampler:
            raise Exception("Passing a sampler isn't supoprt with the CacheDataLoader.")

        if batch_sampler:
            raise Exception("Passing a batch_sampler isn't supoprt with the CacheDataLoader.")

        if isinstance(dataset, IterableDataset):
            raise Exception("Only map-based dataset are supported by the CacheDataLoader for now.")

        cache = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if not cache or len(cache) > 1:
            raise Exception(f"The CacheDataloader should be used with a dataset using a single cache. Found {cache}.")

        cache = cache[0]
        cache._setup(num_workers)
        if not cache.filled and shuffle:
            logger.info("Shuffle is ignored during caching phase")

        distributed_env = _DistributedEnv.detect()
        batch_sampler = CacheBatchSampler(
            len(dataset),
            distributed_env.world_size,
            distributed_env.global_rank,
            num_workers,
            batch_size,
            drop_last,
            shuffle,
            cache,
        )

        super().__init__(
            dataset,
            *args,
            sampler=None,
            batch_sampler=batch_sampler,
            generator=generator,
            collate_fn=CacheCollateFn(),
            num_workers=num_workers,
            **kwargs,
        )

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterPatch(self)
        self.check_worker_number_rationality()
        return _MultiProcessingDataLoaderIterPatch(self)
