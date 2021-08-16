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
from typing import Any, Generator, List, Optional, Tuple

from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException


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
        self.has_raised: bool

        self.reset()

    def setup(self, dataloader: DataLoader, **kwargs) -> None:
        if not isinstance(dataloader, (DataLoader, CombinedLoader)):
            raise MisconfigurationException(
                "The `DataFetcher` should be setup with an instance of a PyTorch ``DataLoader``."
            )
        self.dataloader = dataloader

    def add_batch(self, batch: Any) -> None:
        self.batches.append(batch)

    def fetch_batch(self) -> Any:
        return self.batches.pop(0)

    @property
    def loaders(self) -> List[DataLoader]:
        if not self.dataloader:
            raise MisconfigurationException(
                "The `DataFetcher` should be setup with an instance of a PyTorch ``DataLoader``."
            )
        if isinstance(self.dataloader, CombinedLoader):
            loaders = self.dataloader.loaders
        elif isinstance(self.dataloader, (tuple, list)):
            loaders = self.dataloader
        else:
            loaders = [self.dataloader]
        return loaders

    @property
    def loader_iters(self) -> List[Iterator]:
        if not self.dataloader:
            raise MisconfigurationException(
                "The `DataFetcher` should be setup with an instance of a PyTorch ``DataLoader``."
            )

        if not self.dataloader_iter:
            raise MisconfigurationException("The dataloader_iter isn't available outside the __iter__ context.")

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
        return self.fetching_function()

    def reset(self) -> None:
        self.batches: List = []
        self.fetched: int = 0
        self.done: bool = False


class LightningDataFetcher(AbstractDataFetcher):

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
