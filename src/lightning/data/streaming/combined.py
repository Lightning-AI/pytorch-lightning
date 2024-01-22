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
    """The `CombinedStreamingDataset` enables to stream data from multiple StreamingDataset with the sampling ratio of
    your choice.

    Addtionally, the `CombinedStreamingDataset` keeps track of the number of
    samples fetched to enable resumability of the datasets.

    """

    def __init__(
        self, datasets: List[StreamingDataset], seed: int = 42, weights: Optional[Sequence[float]] = None
    ) -> None:
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        num_datasets = len(datasets)

        if weights is None:
            # Inversely weighted based on length
            self._weights = [1 / float(num_datasets)] * num_datasets
        else:
            self._weights = [w / sum(weights) for w in weights]

        self._iterator: Optional[_CombinedDatasetIterator] = None

    def __len__(self) -> int:
        assert self._weights
        return int(min([1 / w * len(d) for w, d in zip(self._weights, self._datasets) if w > 0]))

    def __iter__(self) -> Iterator[Any]:
        assert self._weights
        self._iterator = _CombinedDatasetIterator(self._datasets, self._seed, self._weights)
        return self._iterator

    def state_dict(self, num_workers: int, batch_size: int) -> Dict[str, Any]:
        if self._iterator is None:
            return {}
        return self._iterator.state_dict(num_workers, batch_size)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if len(state_dict) != len(self._datasets):
            raise RuntimeError(f"The provided state doesn't match the current number of datasets: {self._datasets}.")

        for dataset_idx, dataset in enumerate(self._datasets):
            if str(dataset_idx) not in state_dict:
                raise RuntimeError(f"The provided state doesn't contain the index {dataset_idx}.")

            dataset.load_state_dict(state_dict[str(dataset_idx)])


class _CombinedDatasetIterator(Iterator):
    def __init__(self, datasets: List[StreamingDataset], seed: int, weights: Sequence[float]) -> None:
        self._datasets = datasets
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

    def state_dict(self, num_workers: int = 0, batch_size: int = 1) -> Dict[str, Any]:
        return {
            str(dataset_idx): dataset.state_dict(self._num_samples_yielded[dataset_idx], num_workers, batch_size)
            for dataset_idx, dataset in enumerate(self._datasets)
        }
