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

from time import sleep
import os
from typing import Dict, Iterable, Iterator, Optional, Union
from enum import Enum
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import DataLoader, _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler, Sized

from lightning.data.builder.reader import Reader
from lightning.data.builder.writer import Writer
from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv
import signal
import sys
from torch.utils.data import get_worker_info
from torch._utils import ExceptionWrapper


class Cache:
    def __init__(
        self,
        cache_dir: str,
        data_format: Union[Dict[str, any], str],
        compression: Optional[str] = None,
        chunk_size: int = 2 << 26,
    ):
        super().__init__()
        self._writer = Writer(cache_dir, data_format, chunk_size=chunk_size, compression=compression)
        self._reader = Reader(cache_dir, compression=compression)
        self._cache_dir = cache_dir

        self._env = _DistributedEnv.detect()
        self._worker_env = None
        self._rank = None
        self._dataset_size = None
        self._num_workers = None

    def setup(self, size, num_workers):
        self._dataset_size = size
        self._num_workers = num_workers

    @property
    def rank(self):
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._env.global_rank * self._worker_env.world_size + self._worker_env.rank

        return self._rank

    @property
    def filled(self) -> bool:
        files = os.listdir(self._cache_dir)
        return any(f.endswith("index.json") for f in files)

    def __setitem__(self, index, data):
        self._writer.write(data, self.rank)

    def __getitem__(self, index):
        self._reader.read(index, self.rank)

    def done(self):
        self._writer.done(self.rank)

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


class CacheSampler(Sampler):
    def __init__(self, dataset, generator, shuffle):
        super().__init__(dataset)

        if shuffle:
            self._sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
        else:
            self._sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)


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


class CacheBatchSampler(BatchSampler):
    def __init__(
        self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool, shuffle: bool, cache: Cache
    ):
        super().__init__(sampler, batch_size, drop_last)
        self._cache = cache
        self._shuffle = shuffle

    def __iter__(self):
        if self._cache.filled and self._shuffle:
            return self.__iter__cache__()
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

    def _next_index(self):
        try:
            return super()._next_index()
        except StopIteration as e:
            for worker_queue_idx in range(self._num_workers): 
                self._index_queues[worker_queue_idx].put((worker_queue_idx + self._send_idx, [StopIterationEvent]))
                self._task_info[self._send_idx] = (worker_queue_idx,)

            # Get enough time to receive termination event
            sleep(1)

            raise StopIteration()


class CacheDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        *args,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        shuffle: bool = False,
        generator=None,
        batch_size=None,
        drop_last=False,
        **kwargs
    ):
        if sampler:
            raise Exception("Passing a sampler isn't supoprt with the CacheDataLoader yet.")

        if batch_sampler:
            raise Exception("Passing a batch_sampler isn't supoprt with the CacheDataLoader yet.")

        if isinstance(dataset, IterableDataset):
            raise Exception("Only map-based dataset are supported by the CacheDataLoader for now.")

        cache = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if not cache or len(cache) > 1:
            raise Exception("The CacheDataloader should be used with a dataset using a single cache. Found {cache}.")

        cache = cache[0]
        cache.setup(len(dataset), num_workers)
        batch_sampler = CacheBatchSampler(
            CacheSampler(dataset, generator, shuffle), batch_size, drop_last, shuffle, cache
        )
        super().__init__(
            dataset,
            *args,
            sampler=None,
            batch_sampler=batch_sampler,
            generator=generator,
            collate_fn=CacheCollateFn(),
            **kwargs
        )

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterPatch(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIterPatch(self)
