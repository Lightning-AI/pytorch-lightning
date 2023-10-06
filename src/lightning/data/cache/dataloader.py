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

import asyncio
import inspect
import logging
import os
from importlib import reload
from typing import Any, Callable, Optional

import torch
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

from lightning.data.cache import Cache
from lightning.data.cache.pytree import tree_flatten
from lightning.data.cache.sampler import CacheBatchSampler
from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv

_VIZ_TRACKER_AVAILABLE = RequirementCache("viztracer")

logger = logging.Logger(__name__)

_DEFAULT_CHUNK_BYTES = 1 << 26  # 64M B


def _equal_items(data_1: Any, data_2: Any) -> bool:
    data_1_flattened, _ = tree_flatten(data_1)
    data_2_flattened, _ = tree_flatten(data_2)

    if len(data_1_flattened) != len(data_2_flattened):
        return False

    return all(_equal_item(d1, d2) for d1, d2 in zip(data_1_flattened, data_2_flattened))


def _equal_item(d1, d2) -> bool:
    if not isinstance(d1, type(d2)):
        raise False
    equality = d1 == d2
    if isinstance(equality, torch.Tensor):
        return equality.all()
    return equality


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Optional[str],
        chunk_bytes: Optional[int],
        chunk_size: int,
        compression: Optional[str],
    ):
        self._datataset = dataset
        self._cache = Cache(cache_dir, chunk_bytes=chunk_bytes, chunk_size=chunk_size, compression=compression)
        self._is_deterministic = False

    def __len__(self) -> int:
        return len(self._cache) if self._cache.filled else len(self._datataset)

    def __getitem__(self, index):
        data_1 = self._cache[index] if self._cache.filled else self._datataset[index]
        if not self._cache.filled:
            if not self._is_deterministic:
                data2 = self._datataset[index]
                if not _equal_items(data_1, data2):
                    raise ValueError(
                        f"Your dataset items aren't deterministic. Found {data_1} and {data2} for index {index}."
                        " HINT: Use the `lightning.data.cache.Cache` directly within your dataset."
                    )
                self._is_deterministic = True
            self._cache[index] = data_1
        return data_1


class CacheCollateFn:
    def __init__(self, collate_fn: Optional[Callable] = None):
        self.collate_fn = collate_fn or default_collate

    def __call__(self, items):
        if all(item is None for item in items):
            return None

        # If the __getitem__ method is asynchornous, collect all the items.
        if all(inspect.iscoroutine(item) for item in items):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            items = loop.run_until_complete(asyncio.gather(*items))

        return self.collate_fn(items)


class _SingleProcessDataLoaderIterPatch(_SingleProcessDataLoaderIter):
    """This is overriden to inform the cache is done chunking."""

    def _next_data(self):
        try:
            data = None
            while data is None:
                data = super()._next_data()
            return data
        except StopIteration:
            for v in self._dataset_fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()
                    if not v.filled:
                        v.merge(1)
            raise StopIteration()


class WorkerLoop:
    """Wrap the PyTorch DataLoader WorkerLoop to perform caching and profiling."""

    def __init__(self, global_rank: int, profile: bool = False) -> None:
        self._global_rank = global_rank
        self._profile = profile

    def __call__(self, dataset_kind, *args, **kwargs):
        from torch.utils.data import _DatasetKind
        from torch.utils.data._utils import worker

        from lightning.data.cache.cache import Cache

        rank = _WorkerEnv.detect().rank
        enable_profiling = self._global_rank == 0 and rank == 0 and _VIZ_TRACKER_AVAILABLE and self._profile

        if enable_profiling:
            from viztracer import VizTracer

            tracer = VizTracer(output_file=os.path.join(os.getcwd(), "trace.json"))
            tracer.start()

        # Reload to remove the patching
        reloaded_worker = reload(worker)
        create_fetcher = _DatasetKind.create_fetcher
        fetcher = None

        def create_fetcher_fn(*args, **kwargs):
            nonlocal fetcher
            fetcher = create_fetcher(*args, **kwargs)
            return fetcher

        _DatasetKind.create_fetcher = create_fetcher_fn

        reloaded_worker._worker_loop(dataset_kind, *args, **kwargs)

        if dataset_kind == _DatasetKind.Map:
            for v in fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()

        if enable_profiling:
            tracer.stop()
            tracer.save()


class _MultiProcessingDataLoaderIterPatch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self._cache = loader._cache
        self._num_workers = loader.num_workers
        # Patch PyTorch worker loop to call the `cache.done()` method.
        from torch.utils.data._utils import worker

        worker._worker_loop = WorkerLoop(loader._global_rank, loader._profile)
        super().__init__(loader)

    def _shutdown_workers(self):
        super()._shutdown_workers()
        if not self._cache.filled:
            self._cache.merge(self._num_workers)

    def _next_data(self):
        try:
            data = None
            while data is None:
                data = super()._next_data()
            return data
        except StopIteration as e:
            raise e


class LightningDataLoader(DataLoader):
    __doc__ = DataLoader.__doc__

    def __init__(
        self,
        dataset,
        *args,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        shuffle: bool = False,
        generator=None,
        batch_size=None,
        drop_last=False,
        cache_dir: Optional[str] = None,
        chunk_bytes: Optional[int] = _DEFAULT_CHUNK_BYTES,
        compression: Optional[str] = None,
        profile: bool = False,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        if sampler:
            raise ValueError(
                "The LightningDataLoader relies on its own internal sampler. Passing a sampler isn't supported."
            )

        if batch_sampler:
            raise ValueError(
                "The LightningDataLoader relies on its own internal sampler. Passing a batch_sampler isn't supported."
            )

        if isinstance(dataset, IterableDataset):
            raise ValueError("Only map-based dataset are supported by the LightningDataLoader for now.")

        if profile and not _VIZ_TRACKER_AVAILABLE:
            raise ModuleNotFoundError("To enable DataLoader profiling, run `pip install viztracer`.")

        cache = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if len(cache) > 1:
            raise ValueError(
                "We found several Cache used as attributes from your dataset. Only one is support for now."
            )

        if len(cache) == 0:
            if cache_dir is None:
                raise ValueError("You should provide a `cache_dir` filepath to the LightningDataLoader.")

            dataset = CacheDataset(dataset, cache_dir, chunk_bytes, batch_size, compression)
            cache = dataset._cache
        else:
            cache = cache[0]

        cache._setup(num_workers)

        if not cache.filled and shuffle:
            logger.info("Shuffle is ignored during the caching phase phase")

        self._cache = cache

        distributed_env = _DistributedEnv.detect()
        self._global_rank = distributed_env.global_rank

        batch_sampler = CacheBatchSampler(
            len(dataset),
            distributed_env.world_size,
            self._global_rank,
            num_workers,
            batch_size or 1,
            drop_last,
            shuffle,
            cache,
        )

        self._profile = profile

        super().__init__(
            dataset,
            *args,
            sampler=None,
            batch_sampler=batch_sampler,
            generator=generator,
            collate_fn=CacheCollateFn(collate_fn),
            num_workers=num_workers,
            **kwargs,
        )

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        """Overriden to ensure the `Cache.done()` method is triggered on iteration done."""
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterPatch(self)
        self.check_worker_number_rationality()
        return _MultiProcessingDataLoaderIterPatch(self)
