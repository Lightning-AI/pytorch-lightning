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
from copy import deepcopy
from importlib import reload
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _DatasetKind,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)
from torch.utils.data.sampler import BatchSampler, Sampler

from lightning.data.streaming import Cache
from lightning.data.streaming.combined import (
    __NUM_SAMPLES_YIELDED_KEY__,
    __SAMPLES_KEY__,
    CombinedStreamingDataset,
)
from lightning.data.streaming.constants import _DEFAULT_CHUNK_BYTES, _TORCH_GREATER_EQUAL_2_1_0, _VIZ_TRACKER_AVAILABLE
from lightning.data.streaming.dataset import StreamingDataset
from lightning.data.streaming.sampler import CacheBatchSampler
from lightning.data.utilities.env import _DistributedEnv

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import tree_flatten

logger = logging.Logger(__name__)


def _equal_items(data_1: Any, data_2: Any) -> bool:
    data_1_flattened, _ = tree_flatten(data_1)
    data_2_flattened, _ = tree_flatten(data_2)

    if len(data_1_flattened) != len(data_2_flattened):
        return False

    return all(_equal_item(d1, d2) for d1, d2 in zip(data_1_flattened, data_2_flattened))


def _equal_item(d1: Any, d2: Any) -> bool:
    if not isinstance(d1, type(d2)):
        return False
    equality = d1 == d2
    if isinstance(equality, torch.Tensor):
        return bool(equality.all().item())
    if equality is True:
        return True
    return False


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        cache_dir: str,
        chunk_bytes: Optional[int],
        chunk_size: Optional[int],
        compression: Optional[str],
    ):
        """The `CacheDataset` is a dataset wraper to provide a beginner experience with the Cache.

        Arguments:
            dataset: The dataset of the user
            cache_dir: The folder where the chunks are written to.
            chunk_bytes: The maximal number of bytes to write within a chunk.
            chunk_sie: The maximal number of items to write to a chunk.
            compression: The compression algorithm to use to reduce the size of the chunk.

        """
        self._dataset = dataset
        self._cache = Cache(cache_dir, chunk_bytes=chunk_bytes, chunk_size=chunk_size, compression=compression)
        self._is_deterministic = False

    def __len__(self) -> int:
        return len(self._cache) if self._cache.filled else len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        data_1 = self._cache[index] if self._cache.filled else self._dataset[index]
        if not self._cache.filled:
            if not self._is_deterministic:
                data2 = self._dataset[index]
                if not _equal_items(data_1, data2):
                    raise ValueError(
                        f"Your dataset items aren't deterministic. Found {data_1} and {data2} for index {index}."
                        " HINT: Use the `lightning.data.cache.Cache` directly within your dataset."
                    )
                self._is_deterministic = True
            self._cache[index] = data_1
        return data_1


class CacheCollateFn:
    """This CacheCollateFn is used to accelerate the processing of the data generated using the Cache.

    During the chunking phase, there is no need to return any data from the DataLoader reducing some time.

    Additionally, if the user makes their __getitem__ asynchronous, the collate executes them in parallel.

    """

    def __init__(self, collate_fn: Optional[Callable] = None) -> None:
        self.collate_fn = collate_fn or default_collate

    def __call__(self, items: List[Any]) -> Any:
        if all(item is None for item in items):
            return None

        # If the __getitem__ method is asynchornous, collect all the items.
        if all(inspect.iscoroutine(item) for item in items):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            items = loop.run_until_complete(asyncio.gather(*items))

        return self.collate_fn([item for item in items if item is not None])


class _SingleProcessDataLoaderIterPatch(_SingleProcessDataLoaderIter):
    """This is overriden to inform the cache is done chunking."""

    def _next_data(self) -> Any:
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

    def __call__(
        self,
        dataset_kind: Any,
        dataset: Any,
        index_queue: Any,
        data_queue: Any,
        done_event: Any,
        auto_collation: Any,
        collate_fn: Any,
        drop_last: Any,
        base_seed: Any,
        init_fn: Any,
        worker_id: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from torch.utils.data._utils import worker

        from lightning.data.streaming.cache import Cache

        enable_profiling = self._global_rank == 0 and worker_id == 0 and _VIZ_TRACKER_AVAILABLE and self._profile

        if enable_profiling:
            from viztracer import VizTracer

            tracer = VizTracer(output_file=os.path.join(os.getcwd(), "trace.json"))
            tracer.start()

        # Reload to remove the patching
        reloaded_worker = reload(worker)
        create_fetcher = _DatasetKind.create_fetcher
        fetcher = None

        def create_fetcher_fn(*args: Any, **kwargs: Any) -> "_BaseDatasetFetcher":
            nonlocal fetcher
            fetcher = create_fetcher(*args, **kwargs)
            return fetcher

        _DatasetKind.create_fetcher = create_fetcher_fn  # type: ignore

        reloaded_worker._worker_loop(
            dataset_kind,
            dataset,
            index_queue,
            data_queue,
            done_event,
            auto_collation,
            collate_fn,
            drop_last,
            base_seed,
            init_fn,
            worker_id,
            *args,
            **kwargs,
        )

        if dataset_kind == _DatasetKind.Map:
            assert fetcher
            for v in fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()

        if enable_profiling:
            tracer.stop()
            tracer.save()


class _MultiProcessingDataLoaderIterPatch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader) -> None:
        self._cache = loader._cache
        self._num_workers = loader.num_workers
        # Patch PyTorch worker loop to call the `cache.done()` method.
        from torch.utils.data._utils import worker

        worker._worker_loop = WorkerLoop(loader._global_rank, loader._profile)
        super().__init__(loader)

    def _shutdown_workers(self) -> None:
        super()._shutdown_workers()

        # If the data isn't filled, we trigger an indedm merge
        if not self._cache.filled:
            self._cache.merge(self._num_workers)

    def _next_data(self) -> Any:
        try:
            data = None
            while data is None:
                data = super()._next_data()
            return data
        except StopIteration as e:
            raise e


class CacheDataLoader(DataLoader):
    __doc__ = DataLoader.__doc__

    def __init__(
        self,
        dataset: Any,
        *args: Any,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        cache_dir: Optional[str] = None,
        chunk_bytes: Optional[int] = _DEFAULT_CHUNK_BYTES,
        compression: Optional[str] = None,
        profile: bool = False,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        if sampler:
            raise ValueError(
                "The CacheDataLoader relies on its own internal sampler. Passing a sampler isn't supported."
            )

        if batch_sampler:
            raise ValueError(
                "The CacheDataLoader relies on its own internal sampler. Passing a batch_sampler isn't supported."
            )

        if isinstance(dataset, IterableDataset):
            raise ValueError("Only map-based dataset are supported by the CacheDataLoader for now.")

        if profile and not _VIZ_TRACKER_AVAILABLE:
            raise ModuleNotFoundError("To enable DataLoader profiling, run `pip install viztracer`.")

        cache_list = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if len(cache_list) > 1:
            raise ValueError(
                "We found several Cache used as attributes from your dataset. Only one is support for now."
            )

        if len(cache_list) == 0:
            if cache_dir is None:
                raise ValueError("You should provide a `cache_dir` filepath to the CacheDataLoader.")

            dataset = CacheDataset(dataset, cache_dir, chunk_bytes, batch_size, compression)
            cache = dataset._cache
        else:
            cache = cache_list[0]

        if not cache.filled and shuffle:
            logger.info("Shuffle is ignored during the caching phase phase.")

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
            batch_sampler=batch_sampler,  # type: ignore
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


class _StreamingMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._indexes = (
            list(range(self._loader._latest_worker_idx, self._loader.num_workers))
            if self._loader._latest_worker_idx > 0
            else []
        )
        super().__init__(loader)

    def _try_put_index(self) -> None:
        # Used to restart on the right DataLoader worker
        if self._loader.restore and self._indexes:
            assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

            try:
                index = self._next_index()
            except StopIteration:
                return
            worker_queue_idx = self._indexes.pop(0)

            self._index_queues[worker_queue_idx].put((self._send_idx, index))
            self._task_info[self._send_idx] = (worker_queue_idx,)
            self._tasks_outstanding += 1
            self._send_idx += 1
        else:
            super()._try_put_index()


class StreamingDataLoader(DataLoader):
    """The `StreamingDataLoader` keeps track of the number of samples fetched in order to enable resumability of the
    dataset."""

    __doc__ = DataLoader.__doc__

    def __init__(
        self,
        dataset: Union[StreamingDataset, CombinedStreamingDataset],
        *args: Any,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:  # pyright: ignore
        if not isinstance(dataset, (StreamingDataset, CombinedStreamingDataset)):
            raise RuntimeError(
                "The provided dataset should be either an instance of StreamingDataset or CombinedStreamingDataset."
                f" Found {dataset}."
            )

        self.current_epoch = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._num_samples_yielded_streaming = 0
        self._num_samples_yielded_combined: Dict[int, List[Any]] = {}
        self.rng_state: Optional[Any] = None
        self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
        self._worker_idx_iter: Optional[Any] = None
        self._latest_worker_idx = 0
        self.restore = False
        super().__init__(dataset, *args, batch_size=batch_size, num_workers=num_workers, **kwargs)  # type: ignore

    def __iter__(self) -> Any:
        if not self.restore:
            self._latest_worker_idx = 0
            self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
            self._worker_idx_iter = iter(self._worker_idx)
            self.current_epoch += 1
            self._num_samples_yielded_combined = {}
            self._num_samples_yielded_streaming = 0

        self.dataset.set_epoch(self.current_epoch)

        if isinstance(self.dataset, StreamingDataset):
            assert self.batch_size
            for batch in super().__iter__():
                self._latest_worker_idx = next(self._worker_idx_iter)  # type: ignore
                self._num_samples_yielded_streaming += self.batch_size
                yield batch
        else:
            self.dataset._set_use_streaming_dataloader(True)
            assert self.batch_size
            # TODO: Inject a custom collate function to avoid collating the __NUM_SAMPLES_YIELDED__ key
            for batch in super().__iter__():
                self._latest_worker_idx = next(self._worker_idx_iter)  # type: ignore
                if isinstance(batch, dict) and __NUM_SAMPLES_YIELDED_KEY__ in batch:
                    self._num_samples_yielded_combined[self._latest_worker_idx] = [
                        sample[-1].item() if self.batch_size > 1 else sample.item()
                        for sample in batch[__NUM_SAMPLES_YIELDED_KEY__]
                    ]

                    yield batch[__SAMPLES_KEY__]
                else:
                    yield batch

        self.restore = False

    def state_dict(self) -> Dict[str, Any]:
        if isinstance(self.dataset, StreamingDataset):
            assert self.batch_size
            return {
                "dataset": self.dataset.state_dict(
                    self._num_samples_yielded_streaming, self.num_workers, self.batch_size
                ),
                "current_epoch": self.current_epoch,
                "num_samples_yielded": self._num_samples_yielded_streaming,
                "latest_worker_idx": self._latest_worker_idx,
            }

        num_samples_yieled = [0 for _ in range(len(list(self._num_samples_yielded_combined.values())[0]))]
        for worker_idx in self._num_samples_yielded_combined:
            for dataset_idx, samples_yieled in enumerate(self._num_samples_yielded_combined[worker_idx]):
                num_samples_yieled[dataset_idx] += samples_yieled

        return {
            "dataset": self.dataset.state_dict(self.num_workers, self.batch_size, num_samples_yieled),
            "current_epoch": self.current_epoch if self.restore else self.current_epoch - 1,
            "latest_worker_idx": self._latest_worker_idx,
            "num_samples_yielded": deepcopy(self._num_samples_yielded_combined),
        }

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Any): The state.

        """
        self.current_epoch = obj["current_epoch"]

        if isinstance(self.dataset, StreamingDataset):
            self._num_samples_yielded_streaming = obj["num_samples_yielded"]
        else:
            self._num_samples_yielded_combined = obj["num_samples_yielded"]

        # Used to restart on the next DataLoader worker from the previous run.
        self._latest_worker_idx = obj["latest_worker_idx"] + 1
        self._worker_idx_iter = iter(self._worker_idx)
        for _ in range(self._latest_worker_idx):
            next(self._worker_idx_iter)

        # Inform we are resuming and disable resetting the StreamingDataLoader state.
        # This is toggle back to False when the `__iter__` method of the StreamingDataLoader completes.
        self.restore = True

        if isinstance(self.dataset, CombinedStreamingDataset):
            self.dataset._set_use_streaming_dataloader(True)
            self.dataset.load_state_dict(obj)
        elif isinstance(self.dataset, StreamingDataset):
            self.dataset.load_state_dict(obj["dataset"])
        else:
            raise RuntimeError("The provided dataset should be a `StreamingDataset` or a `CombinedStreamingDataset`.")

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        """Overriden to ensure the `Cache.done()` method is triggered on iteration done."""
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        self.check_worker_number_rationality()
        return _StreamingMultiProcessingDataLoaderIter(self)
