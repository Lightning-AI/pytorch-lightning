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

from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

from lightning.data.cache import Cache
from lightning.data.cache.sampler import CacheBatchSampler
from lightning.data.datasets.env import _DistributedEnv

logger = logging.Logger(__name__)


class CacheCollateFn:
    def __init__(self):
        self.collate_fn = default_collate

    def __call__(self, items):
        if all(item is None for item in items):
            return None
        return self.collate_fn(items)


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
            raise Exception("Passing a sampler isn't supported with the CacheDataLoader.")

        if batch_sampler:
            raise Exception("Passing a batch_sampler isn't supported with the CacheDataLoader.")

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
