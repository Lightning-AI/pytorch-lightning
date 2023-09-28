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
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
    DataLoader,
)
from torch.utils.data.sampler import BatchSampler, Sampler, SequentialSampler, Sized

from lightning.data.cache.reader import BinaryReader
from lightning.data.cache.writer import BinaryWriter

logger = logging.Logger(__name__)


class Cache:
    def __init__(
        self,
        cache_dir: str,
        data_format: Union[Dict[str, any], str],
        compression: Optional[str] = None,
        chunk_size: int = 2 << 26,
    ):
        super().__init__()
        self._writer = BinaryWriter(cache_dir, data_format, chunk_size=chunk_size, compression=compression)
        self._reader = BinaryReader(cache_dir, compression=compression)
        self._cache_dir = cache_dir

    # TODO: Find a way to make this faster
    @property
    def filled(self) -> bool:
        files = os.listdir(self._cache_dir)
        return any(f.endswith("index.json") for f in files)

    def __setitem__(self, index, data):
        self._writer[index] = data

    def __getitem__(self, index):
        return self._reader.read(index)

    def done(self):
        self._writer.done()

    def __len__(self):
        return self._reader.get_length()

    def get_chunk_interval(self):
        return self._reader.get_chunk_interval()


class _SingleProcessDataLoaderIterPatch(_SingleProcessDataLoaderIter):
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
        super().__init__(None)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.indices = range(dataset_size)
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

    @property
    def done(self) -> bool:
        return len(self._done) == len(self.iterators)

    def __iter__(self):
        self._done = set()

        for sampler in self.samplers:
            self.iterators.append(iter(sampler))

        return self

    def __next__(self):
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
        self, dataset_size: int, num_workers: int, batch_size: int, drop_last: bool, shuffle: bool, cache: Cache
    ):
        if num_workers >= 1:
            sampler = CacheSampler(dataset_size, num_workers, batch_size)
        else:
            sampler = SequentialSampler(range(dataset_size))
        super().__init__(sampler, batch_size, drop_last)
        self._cache = cache
        self._shuffle = shuffle
        self._num_workers = num_workers

    def __modified_iter__(self) -> Iterator[List[int]]:
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
            return self.__iter__cache__()
        if self._num_workers >= 1:
            return self.__modified_iter__()
        return super().__iter__()

    def __iter__cache__(self):
        chunk_intervals = self._cache.get_chunk_interval()[:-1]
        shuffled_chunk_intervals = np.random.permutation(chunk_intervals)

        dataset = []
        for interval in shuffled_chunk_intervals:
            interval_indices = np.arange(interval[0], interval[1])
            shuffled_interval_indices = np.random.permutation(interval_indices)
            dataset.extend(shuffled_interval_indices.tolist())

        if len(dataset) != len(self.sampler):
            raise Exception("The generated indices don't match the initial length of the sampler.")

        self.sampler = IteratorSampler(dataset)

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


StopIterationEvent = "StopIterationEvent"


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
            raise Exception("Passing a sampler isn't supoprt with the CacheDataLoader yet.")

        if batch_sampler:
            raise Exception("Passing a batch_sampler isn't supoprt with the CacheDataLoader yet.")

        if isinstance(dataset, IterableDataset):
            raise Exception("Only map-based dataset are supported by the CacheDataLoader for now.")

        cache = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if not cache or len(cache) > 1:
            raise Exception(f"The CacheDataloader should be used with a dataset using a single cache. Found {cache}.")

        cache = cache[0]
        if not cache.filled and shuffle:
            logger.info("Shuffle is ignored during caching phase")

        super().__init__(
            dataset,
            *args,
            sampler=None,
            batch_sampler=CacheBatchSampler(len(dataset), num_workers, batch_size, drop_last, shuffle, cache),
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
