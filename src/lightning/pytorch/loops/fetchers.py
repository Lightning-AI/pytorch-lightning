# Copyright The Lightning AI team.
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

from typing import Any, Iterable, Iterator, List, Optional, Sized, Tuple, Union

from torch.utils.data.dataloader import DataLoader

from lightning.fabric.utilities.data import has_len
from lightning.pytorch.utilities.combined_loader import (
    _Sequential,
    _shutdown_workers_and_reset_iterator,
    CombinedLoader,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def _profile_nothing() -> None:
    pass


class _DataFetcher(Iterator):
    def __init__(self) -> None:
        self._dataloader: Optional[Iterable] = None
        self.dataloader_iter: Optional[Iterator] = None
        self.fetched: int = 0
        self.done: bool = False
        self._start_profiler = _profile_nothing
        self._stop_profiler = _profile_nothing

    def setup(self, dataloader: Iterable) -> None:
        self._dataloader = dataloader

    @property
    def dataloader(self) -> Iterable:
        if self._dataloader is None:
            raise MisconfigurationException(
                f"`{self.__class__.__name__}` should have been `setup` with a dataloader iterable."
            )
        return self._dataloader

    def __iter__(self) -> "_DataFetcher":
        self.reset()
        self.dataloader_iter = iter(self.dataloader)
        return self

    def __next__(self) -> Any:
        self._start_profiler()
        assert self.dataloader_iter is not None
        try:
            data = next(self.dataloader_iter)
        except StopIteration as e:
            self.done = True
            raise e
        finally:
            self._stop_profiler()
        self.fetched += 1
        return data

    def reset(self) -> None:
        self.fetched = 0
        self.done = False

    def teardown(self) -> None:
        self.reset()
        if isinstance(self._dataloader, CombinedLoader):
            self._dataloader.reset()
        if isinstance(self._dataloader, DataLoader):
            _shutdown_workers_and_reset_iterator(self._dataloader)
        self.dataloader_iter = None


class _PrefetchDataFetcher(_DataFetcher):
    """This class is used to control batch fetching flow.

    Args:
        prefetch_batches: Number of batches to pre-fetch. Pre-fetching at least 1 batch is necessary to properly track
            whether a batch is the last one (available with :attr:`self.done`) when the length is not available.
    """

    def __init__(self, prefetch_batches: int = 1) -> None:
        super().__init__()
        if prefetch_batches < 0:
            raise ValueError("`prefetch_batches` should at least be 0.")
        self.prefetch_batches = prefetch_batches
        self.batches: List[Any] = []
        self._has_len = False

    def setup(self, dataloader: Iterable) -> None:
        super().setup(dataloader)
        self._has_len = has_len(dataloader)

    def __iter__(self) -> "_PrefetchDataFetcher":
        super().__iter__()
        if self._has_len:
            # ignore pre-fetching, it's not necessary
            return self
        # prefetch batches to know when the iterator will be exhausted in advance
        iterator = self.dataloader_iter
        assert iterator is not None
        for _ in range(self.prefetch_batches):
            try:
                self._fetch_next_batch(iterator)
            except StopIteration:
                # this would only happen when prefetch_batches > the number of batches available and makes
                # `__next__` jump directly to the empty iterator case without trying to fetch again
                self.done = True
                break
        return self

    def __next__(self) -> Any:
        assert self.dataloader_iter is not None
        if self.batches:
            # there are pre-fetched batches already from a previous `prefetching` call.
            # consume one
            batch = self.batches.pop(0)
            try:
                # refill the consumed batch
                self._fetch_next_batch(self.dataloader_iter)
            except StopIteration:
                # no more batches to fetch. we are done only if all pre-fetched batches were returned
                self.done = not self.batches
        elif not self.done:
            # this will run only when no pre-fetching was done.
            try:
                self._fetch_next_batch(self.dataloader_iter)
                # consume the batch we just fetched
                batch = self.batches.pop(0)
            except StopIteration as e:
                self.done = True
                raise e
        else:
            # the iterator is empty
            raise StopIteration
        return batch

    def _fetch_next_batch(self, iterator: Iterator) -> None:
        self._start_profiler()
        try:
            batch = next(iterator)
        finally:
            self._stop_profiler()
        self.fetched += 1
        if self._has_len:
            # when we don't prefetch but the dataloader is sized, we use the length for `done`
            dataloader = self.dataloader
            assert isinstance(dataloader, Sized)  # `_has_len` is True
            self.done = self.fetched >= len(dataloader)
        self.batches.append(batch)

    def reset(self) -> None:
        super().reset()
        self.batches = []


class _DataLoaderIterDataFetcher(_DataFetcher):
    """This class is used to return directly the `dataloader_iter` to the ``LightningModule`` training_step for
    users to implement their own pre-fetching logic. This feature can be activated as follows:

    Example::

        Class MyModel(LightningModule):
            def training_step(self, dataloader_iter: Iterator, batch_idx: int) -> None:
                # it is the user responsibility to fetch and move the batch to the right device.
                batch = next(dataloader_iter)
                batch = batch.to(self.device)
                ...
    """

    def __iter__(self) -> "_DataLoaderIterDataFetcher":
        super().__iter__()
        self.iterator = iter(_DataFetcherWrapper(self))
        return self

    def __next__(self) -> Union["_DataFetcherWrapper", Tuple["_DataFetcherWrapper", int, int]]:
        if self.done:
            raise StopIteration
        assert isinstance(self.iterator, _DataFetcherWrapper)
        if self._is_sequential:
            sequential_mode = self.dataloader._iterator
            assert isinstance(sequential_mode, _Sequential)
            batch_idx = sequential_mode._idx
            dataloader_idx = sequential_mode._iterator_idx
            return self.iterator, batch_idx, dataloader_idx
        return self.iterator

    @property
    def _is_sequential(self) -> bool:
        return isinstance(self.dataloader, CombinedLoader) and self.dataloader._mode == "sequential"


class _DataFetcherWrapper(Iterator):
    def __init__(self, data_fetcher: _DataLoaderIterDataFetcher) -> None:
        self.data_fetcher = data_fetcher

    def __next__(self) -> Any:
        out = super(_DataLoaderIterDataFetcher, self.data_fetcher).__next__()
        if self.data_fetcher._is_sequential:
            # avoid breaking change with sequential mode and dataloader_iter. this is okay because
            # dataloader_iter + sequential + multiple dataloaders is not supported so the `*_step(..., batch_idx)` value
            # and the batch_index we are excluding here will match
            return out[0]
        return out
