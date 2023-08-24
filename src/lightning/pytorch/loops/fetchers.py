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

from typing import Any, Iterator, List, Optional, Tuple, Union

from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities.combined_loader import _Sequential, CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def _profile_nothing() -> None:
    pass


class _DataFetcher(Iterator):
    def __init__(self) -> None:
        self._combined_loader: Optional[CombinedLoader] = None
        self.iterator: Optional[Iterator] = None
        self.fetched: int = 0
        self.done: bool = False
        self.length: Optional[int] = None
        self._start_profiler = _profile_nothing
        self._stop_profiler = _profile_nothing

    @property
    def combined_loader(self) -> CombinedLoader:
        if self._combined_loader is None:
            raise MisconfigurationException(
                f"`{self.__class__.__name__}` should have been `setup` with a `CombinedLoader`."
            )
        return self._combined_loader

    def setup(self, combined_loader: CombinedLoader) -> None:
        self._combined_loader = combined_loader
        self.length = sized_len(combined_loader)
        self.done = self.length == 0

    def __iter__(self) -> "_DataFetcher":
        self.reset()
        self.iterator = iter(self.combined_loader)
        return self

    def __next__(self) -> Any:
        assert (iterator := self.iterator) is not None
        self._start_profiler()
        try:
            batch = next(iterator)
        except StopIteration:
            self.done = True
            raise
        finally:
            self._stop_profiler()
        self.fetched += 1
        if self.length is not None:
            self.done = self.fetched >= self.length
        return batch

    def reset(self) -> None:
        self.fetched = 0
        self.done = False

    def teardown(self) -> None:
        self.reset()
        if self._combined_loader is not None:
            self._combined_loader.reset()
        self.iterator = None


class _PrefetchDataFetcher(_DataFetcher):
    """This class is used to control batch fetching flow.

    Args:
        prefetch_batches: Number of batches to pre-fetch. Pre-fetching at least 1 batch is necessary to properly track
            whether a batch is the last one (available with :attr:`self.done`) when the length is not available. The
            value of this argument is ignored when the length is available.

    """

    def __init__(self, prefetch_batches: int = 1) -> None:
        super().__init__()
        if prefetch_batches < 0:
            raise ValueError("`prefetch_batches` should at least be 0.")
        self.prefetch_batches = prefetch_batches
        self.batches: List[Any] = []

    def __iter__(self) -> "_PrefetchDataFetcher":
        super().__iter__()
        if self.length is not None:
            # ignore pre-fetching, it's not necessary
            return self
        # prefetch batches to know when the iterator will be exhausted in advance
        for _ in range(self.prefetch_batches):
            try:
                batch = super().__next__()
                self.batches.append(batch)
            except StopIteration:
                # this would only happen when prefetch_batches > the number of batches available and makes
                # `__next__` jump directly to the empty iterator case without trying to fetch again
                break
        return self

    def __next__(self) -> Any:
        if self.batches:
            # there are pre-fetched batches already from a previous `prefetching` call.
            # consume one
            batch = self.batches.pop(0)
            try:
                # refill the consumed batch
                self.batches.append(super().__next__())
            except StopIteration:
                # no more batches to fetch. we are done only if all pre-fetched batches were returned
                self.done = not self.batches
        elif not self.done:
            # this will run only when no pre-fetching was done.
            batch = super().__next__()
        else:
            # the iterator is empty
            raise StopIteration
        return batch

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
        self.iterator_wrapper = iter(_DataFetcherWrapper(self))
        return self

    def __next__(self) -> Union["_DataFetcherWrapper", Tuple["_DataFetcherWrapper", int, int]]:
        if self.done:
            raise StopIteration
        assert isinstance(self.iterator_wrapper, _DataFetcherWrapper)
        if self._is_sequential:
            mode = self.combined_loader._iterator
            assert isinstance(mode, _Sequential)
            batch_idx = mode._idx
            dataloader_idx = mode._iterator_idx
            return self.iterator_wrapper, batch_idx, dataloader_idx
        return self.iterator_wrapper

    @property
    def _is_sequential(self) -> bool:
        return self.combined_loader._mode == "sequential"


class _DataFetcherWrapper(Iterator):
    def __init__(self, data_fetcher: _DataLoaderIterDataFetcher) -> None:
        self.data_fetcher = data_fetcher

    @property
    def done(self) -> bool:
        return self.data_fetcher.done

    @property
    def fetched(self) -> int:
        return self.data_fetcher.fetched

    @property
    def length(self) -> Optional[int]:
        return self.data_fetcher.length

    def __next__(self) -> Any:
        out = super(_DataLoaderIterDataFetcher, self.data_fetcher).__next__()
        if self.data_fetcher._is_sequential:
            # avoid breaking change with sequential mode and dataloader_iter. this is okay because
            # dataloader_iter + sequential + multiple dataloaders is not supported so the `*_step(..., batch_idx)` value
            # and the batch_index we are excluding here will match
            return out[0]
        return out
