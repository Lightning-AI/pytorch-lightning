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
from datetime import datetime
from typing import Any, Optional

import torch
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
from lightning.data.datasets.env import _DistributedEnv

logger = logging.Logger(__name__)


def _equal_items(data_1: Any, data_2: Any) -> bool:
    data_1_flattened, _ = tree_flatten(data_1)
    data_2_flattened, _ = tree_flatten(data_2)

    if len(data_1_flattened) != len(data_2_flattened):
        return False

    return all(_equal_item(d1, d2) for d1, d2 in zip(data_1_flattened, data_2_flattened))


def _equal_item(d1, d2) -> bool:
    if not isinstance(d1, type(d2)):
        raise False
    if isinstance(d1, torch.Tensor) and isinstance(d2, torch.Tensor):
        return torch.equal(d1, d2)
    return d1 == d2


class CacheDataset(Dataset):
    def __init__(
        self, dataset: Dataset, cache_dir: Optional[str], chunk_size: Optional[int], compression: Optional[str]
    ):
        self._datataset = dataset
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "cache_dir", datetime.now().strftime("%m-%d-%Y-%H-%M"))
            os.makedirs(cache_dir, exist_ok=True)
            chunk_size = 2 << 26
        self.cache = Cache(cache_dir, chunk_size=chunk_size, compression=compression)
        self.is_deterministic = False

    def __len__(self) -> int:
        return len(self.cache) if self.cache.filled else len(self._datataset)

    def __getitem__(self, index):
        data_1 = self.cache[index] if self.cache.filled else self._datataset[index]
        if not self.cache.filled:
            if not self.is_deterministic:
                data2 = self._datataset[index]
                if not _equal_items(data_1, data2):
                    raise ValueError(
                        f"Your dataset items aren't deterministic. Found {data_1} and {data2} for index {index}."
                        " HINT: Use the `lightning.data.cache.Cache` directly within your dataset."
                    )
                self.is_deterministic = True
            self.cache[index] = data_1
        return data_1


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


class LightningDataLoader(DataLoader):
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
        cache_dir: Optional[str] = None,
        chunk_size: Optional[int] = 2 << 26,
        compression: Optional[str] = None,
        **kwargs,
    ):
        if sampler:
            raise Exception(
                "The LightningDataLoader relies on its own internal sampler. Passing a sampler isn't supported."
            )

        if batch_sampler:
            raise Exception(
                "The LightningDataLoader relies on its own internal sampler. Passing a batch_sampler isn't supported."
            )

        if isinstance(dataset, IterableDataset):
            raise Exception("Only map-based dataset are supported by the LightningDataLoader for now.")

        cache = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if len(cache) > 1:
            raise Exception("We found several Cache used as attributes from your dataset. Only one is support for now.")

        if len(cache) == 0:
            if cache_dir is None:
                logger.info("You can provide a `cache_dir` filepath to the LightningDataLoader.")

            dataset = CacheDataset(dataset, cache_dir, chunk_size, compression)
            cache = dataset.cache
        else:
            cache = cache[0]

        cache._setup(num_workers)

        if not cache.filled and shuffle:
            logger.info("Shuffle is ignored during the caching phase phase")

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
        """Overriden to ensure the `Cache.done()` method is triggered on iteration done."""
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterPatch(self)
        self.check_worker_number_rationality()
        return _MultiProcessingDataLoaderIterPatch(self)
