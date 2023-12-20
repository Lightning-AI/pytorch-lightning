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

import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

from torch.utils.data import IterableDataset

from lightning.data.streaming.dataset import StreamingDataset


class CombinedStreamingDataset(IterableDataset):
    def __init__(
        self, datasets: List[StreamingDataset], seed: int = 42, weights: Optional[Sequence[float]] = None
    ) -> None:
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        num_datasets = len(datasets)

        # Inversely weighted based on length
        if weights is None:
            self._weights = [1 / float(num_datasets)] * num_datasets
        else:
            self._weights = [w / sum(weights) for w in weights]

        self._iterator: Optional[CombinedDatasetIterator] = None

    def __iter__(self) -> Iterator[Any]:
        assert self._weights
        self._iterartor = CombinedDatasetIterator(self._datasets, self._seed, self._weights)
        return self._iterartor

    def state_dict(self) -> Optional[Dict[str, Any]]:
        if self._iterartor is None:
            return {}


class CombinedDatasetIterator(Iterator):
    def __init__(self, datasets: List[StreamingDataset], seed: int, weights: Sequence[float]) -> None:
        self._dataset_iters = [iter(dataset) for dataset in datasets]
        self._dataset_indexes = list(range(len(datasets)))
        self._num_samples_yielded = [0 for _ in range(len(datasets))]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self) -> Any:
        # randomly select a dataset index
        (dataset_index,) = self._rng.choices(self._dataset_indexes, weights=self._weights, k=1)

        # keep track the sample was fetched
        self._num_samples_yielded[dataset_index] += 1

        # return a new sample
        return next(self._dataset_iters[dataset_index])
