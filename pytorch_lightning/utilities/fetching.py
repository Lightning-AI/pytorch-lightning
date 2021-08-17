# Copyright The PyTorch Lightning team.
#
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
from collections.abc import Iterable, Iterator
from copy import deepcopy
from functools import partial
from typing import Any, Generator, List, Optional, Tuple

from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator
from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections
from pytorch_lightning.utilities.auto_restart import (
    _add_capture_metadata_collate,
    IteratorState,
    MergedIteratorState,
    patch_dataloader_iterator,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _FAULT_TOLERANT_ENABLED


class AbstractDataFetcher(ABC):

    """
    This class is used to control batch fetching flow.
    """

    @abstractmethod
    def fetching_function(self) -> Generator:
        pass

    def __init__(
        self,
        prefetch_batches: int = 0,
    ) -> None:
        if not isinstance(prefetch_batches, int) or (isinstance(prefetch_batches, int) and prefetch_batches < 0):
            raise MisconfigurationException("`prefetch_batches` should at least be 0.")

        self.prefetch_batches = prefetch_batches + 1

        self.dataloader: Optional[Iterable] = None
        self.dataloader_iter: Optional[Iterator] = None

        self.batches: List
        self.fetched: int
        self.done: bool

        self.reset()

    def setup(self, dataloader: DataLoader, **kwargs) -> None:
        self._add_capture_metadata_collate(dataloader)

        self.dataloader = dataloader
        if isinstance(dataloader, DataLoader) and not isinstance(dataloader.collate_fn, partial):
            _add_capture_metadata_collate(dataloader)

    @staticmethod
    def _add_capture_metadata_collate(dataloader: Iterable) -> None:
        if not isinstance(dataloader, (DataLoader, CombinedLoader)):
            return

        if isinstance(dataloader, CombinedLoader):
            dataloader = dataloader.loaders

        apply_to_collection(dataloader, DataLoader, _add_capture_metadata_collate)

    def add_batch(self, batch) -> None:
        self.batches.append(batch)

    def fetch_batch(self) -> Any:
        return self.batches.pop(0)

    def _apply_patch(self):
        def _apply_patch_fn(loader: DataLoader, iterator: Iterator):
            if isinstance(loader, CycleIterator):
                loader = loader.loader
                # cycle_iterator = iterator
                iterator = iterator._loader_iter

            if isinstance(loader, DataLoader) and _FAULT_TOLERANT_ENABLED:
                loader._lightning_fetcher = self
                patch_dataloader_iterator(loader, iterator, self)

        apply_to_collections(self.loaders, self.loader_iters, (Iterator, DataLoader), _apply_patch_fn)

    def _store_dataloader_iter_state(
        self, dataloader_iter: Iterator, dataloader_iter_states: List[IteratorState]
    ) -> None:
        if getattr(dataloader_iter, "cache_states", None) is None:
            dataloader_iter.cache_states = {}

        if getattr(dataloader_iter, "state", None) is None:
            dataloader_iter.state = MergedIteratorState()

        for iter_state in dataloader_iter_states:
            iter_name = iter_state.name
            if iter_name not in dataloader_iter.cache_states:
                dataloader_iter.cache_states[iter_name] = []
            dataloader_iter.cache_states[iter_name].append(iter_state)

        if self.fetched >= self.prefetch_batches:
            for iter_state in dataloader_iter_states:
                if len(dataloader_iter.state):
                    dataloader_iter.previous_state = deepcopy(dataloader_iter.state)
                iter_name = iter_state.name
                state = dataloader_iter.cache_states[iter_name].pop(0)
                dataloader_iter.state.update(iter_name, state)

    @property
    def loaders(self) -> List[DataLoader]:
        if self.dataloader is None:
            raise MisconfigurationException(
                "The `DataFetcher` should be setup with an instance of a PyTorch ``DataLoader``."
            )
        if isinstance(self.dataloader, CombinedLoader):
            loaders = self.dataloader.loaders
        else:
            loaders = [self.dataloader]
        return loaders

    @property
    def loader_iters(self) -> List[Iterator]:
        if self.dataloader is None:
            raise MisconfigurationException(
                "The `DataFetcher` should be setup with an instance of a PyTorch ``DataLoader``."
            )

        if self.dataloader_iter is None:
            raise MisconfigurationException("The `dataloader_iter` isn't available outside the __iter__ context.")

        if isinstance(self.dataloader, CombinedLoader):
            loader_iters = self.dataloader_iter.loader_iters
        else:
            loader_iters = [self.dataloader_iter]
        return loader_iters

    @property
    def state(self) -> Any:
        def collect_state(iterator: Iterator):
            return iterator.state

        return apply_to_collection(self.loader_iters, Iterator, collect_state)

    def __iter__(self) -> Generator[Tuple[Any, bool], None, None]:
        if self.dataloader is None:
            raise MisconfigurationException("The iterate hasn't been provided. HINT: Did you call setup function ?.")
        self.reset()
        self.dataloader_iter = iter(self.dataloader)
        self._apply_patch()
        return self.fetching_function()

    def reset(self) -> None:
        self.batches: List = []
        self.dataloader: Optional[Iterable]
        self.fetched: int = 0
        self.done: bool = False


class DataFetcher(AbstractDataFetcher):

    """
    This class is used to control batch fetching flow.
    """

    def fetching_function(self) -> Generator:
        self.done = False
        while not self.done:
            self._prefetching(self.prefetch_batches)

            for batch in self.dataloader_iter:
                yield_batch = self.fetch_batch()
                self.add_batch(batch)
                self.fetched += 1
                # yield last and has next
                yield yield_batch, False

            yield from self._consume_prefetched_batches()

    def _consume_prefetched_batches(self) -> Generator:
        self.done = True
        while self.batches:
            if len(self.batches) == 1:
                yield self.batches.pop(0), True
            else:
                yield self.batches.pop(0), False

    def _prefetching(self, prefetch_batches: int) -> None:
        for _ in range(prefetch_batches):
            try:
                batch = next(self.dataloader_iter)
                self.fetched += 1
                self.add_batch(batch)
            except StopIteration:
                break
