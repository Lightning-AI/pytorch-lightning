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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

logger = logging.Logger(__name__)


@dataclass
class ChunkedIndex:
    index: int
    chunk_index: int
    chunk_indexes: Optional[List[int]] = None
    last_index: bool = False


class CacheBatchSampler:
    def __init__(
        self,
        dataset_size: int,
        num_replicas: int,
        global_rank: int,
        num_workers: int,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        cache: Any,
    ):
        """The CacheBatchSampler handles the generation of batch indices.

        If the cache isn't filled, the batch sampler alternates with ordered indices for the writer to chunk the dataset
        If the cache is filled, it acts as normal BatchSampler.

        Arguments:
            dataset_size: The size of the dataset.
            num_replicas: The number of processes involves in the distributed training.
            global_rank: The global_rank of the given process
            num_workers: The number of workers provided to the DataLoader.
            batch_size: The number of items in a batch.
            drop_last: Whether to drop the last batch of data.
            shuffle: Whether the data should be shuffled.
            cache: The cache associated to the dataset.

        """
        self._dataset_size = dataset_size
        self._num_replicas = num_replicas
        self._global_rank = global_rank
        self._cache = cache
        self._shuffle = shuffle
        self._num_workers = num_workers or 1
        self._shuffled_chunk_intervals = None
        self._batch_size = batch_size

        self._drop_last = drop_last
        self._length = 0

        # Before starting, ensures the chunk indices are properly defined.
        self._validate()

    def _validate(self) -> None:
        """Checks each worker is getting sucessive indices."""
        if self._num_workers > 1 and not self._cache.filled:
            batches: Dict[int, Any] = {}
            for batch_index, batch_indices in enumerate(self):
                self._length += 1
                worker_index = batch_index % self._num_workers
                if worker_index not in batches:
                    batches[worker_index] = []
                    batches[worker_index].extend(batch_indices)
                elif len(batch_indices) > 0:
                    batches[worker_index].extend(batch_indices)

            for indices in batches.values():
                indices = np.asarray(indices)
                diff = indices[1:] - (indices[:-1] + 1)
                if diff.sum() != 0:
                    raise RuntimeError("This shouldn't have happened. There is a bug in the CacheSampler.")

    def __iter__(self) -> Iterator[List[Union[int, ChunkedIndex]]]:
        # When the cache is filled, we need to iterate though the chunks
        if self._cache.filled:
            if self._num_replicas == 1:
                return self.__iter_from_chunks_non_distributed__()
            return self.__iter_from_chunks_distributed__()

        # shuffle is ignored while building the binarized version of the dataset
        if self._num_replicas == 1:
            return self.__iter_non_distributed__()
        return self.__iter_distributed__()

    def __iter_non_distributed__(self) -> Iterator[List[Union[int, ChunkedIndex]]]:
        worker_size = self._dataset_size // self._num_workers
        indices = list(range(self._dataset_size))
        worker_indices = []
        for worker_idx in range(self._num_workers):
            is_last = worker_idx == self._num_workers - 1
            start = worker_idx * worker_size
            end = self._dataset_size if is_last else (worker_idx + 1) * worker_size
            worker_indices.append(indices[start:end])

        assert sum([len(s) for s in worker_indices]) == self._dataset_size

        worker_indices_batches = [self._chunk_list(indices, self._batch_size) for indices in worker_indices]

        yield from self.__iter_indices_per_workers__(worker_indices_batches)

    def __iter_distributed__(self) -> Iterator[List[Union[int, ChunkedIndex]]]:
        self.indices = list(range(self._dataset_size))
        replica_size = self._dataset_size // self._num_replicas
        worker_size = self._dataset_size // (self._num_replicas * self._num_workers)
        for rank in range(self._num_replicas):
            if rank != self._global_rank:
                continue

            is_last_replica = rank == self._num_replicas - 1
            start_replica = rank * replica_size
            end_replica = self._dataset_size if is_last_replica else (rank + 1) * replica_size
            replica_indices = self.indices[start_replica:end_replica]

            replica_size = len(replica_indices)

            worker_indices = []
            for worker_idx in range(self._num_workers):
                is_last_worker = worker_idx == self._num_workers - 1
                start_worker = worker_idx * worker_size
                end_worker = replica_size if is_last_worker else (worker_idx + 1) * worker_size
                worker_indices.append(replica_indices[start_worker:end_worker])

        assert sum([len(s) for s in worker_indices]) == len(replica_indices)

        worker_indices_batches = [self._chunk_list(indices, self._batch_size) for indices in worker_indices]

        yield from self.__iter_indices_per_workers__(worker_indices_batches)

    def __iter_from_chunks_non_distributed__(self) -> Iterator[List[Union[int, ChunkedIndex]]]:
        chunk_intervals = self._cache.get_chunk_intervals()
        shuffled_indexes = np.random.permutation(range(len(chunk_intervals)))
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes]
        yield from self.__iter_from_shuffled_chunks(shuffled_indexes.tolist(), shuffled_chunk_intervals)

    def __iter_from_chunks_distributed__(self) -> Iterator[List[Union[int, ChunkedIndex]]]:
        chunk_intervals = self._cache.get_chunk_intervals()
        shuffled_indexes = np.random.permutation(range(len(chunk_intervals)))
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes]

        replica_chunks = []
        replica_intervals = []
        for index, (chunk_index, chunk_interval) in enumerate(zip(shuffled_indexes, shuffled_chunk_intervals)):
            if index % self._num_replicas == self._global_rank:
                replica_chunks.append(chunk_index)
                replica_intervals.append(chunk_interval)

        yield from self.__iter_from_shuffled_chunks(replica_chunks, replica_intervals)

    def __iter_from_shuffled_chunks(
        self, shuffled_indexes: List[int], shuffled_chunk_intervals: List[List[int]]
    ) -> Iterator[List[Union[int, ChunkedIndex]]]:
        chunks_per_workers: List[List[int]] = [[] for _ in range(self._num_workers)]
        for i, chunk_index in enumerate(shuffled_indexes):
            chunks_per_workers[i % self._num_workers].append(chunk_index)

        indices_per_workers: List[List[ChunkedIndex]] = [[] for _ in range(self._num_workers)]

        for i, (chunk_index, chunk_interval) in enumerate(zip(shuffled_indexes, shuffled_chunk_intervals)):
            worker_id = i % self._num_workers
            interval_indices = np.arange(chunk_interval[0], chunk_interval[1])
            shuffled_interval_indices = np.random.permutation(interval_indices).tolist()
            is_empty = len(indices_per_workers[worker_id]) == 0
            indices_per_workers[worker_id].extend(
                [
                    ChunkedIndex(
                        index,
                        chunk_index,
                        chunk_indexes=chunks_per_workers[worker_id] if j == 0 and is_empty else None,
                    )
                    for j, index in enumerate(shuffled_interval_indices)
                ]
            )

        indices_per_workers_splitted = [self._chunk_list(indices, self._batch_size) for indices in indices_per_workers]

        yield from self.__iter_indices_per_workers__(indices_per_workers_splitted)

    def __len__(self) -> int:
        return self._length

    def __iter_indices_per_workers__(
        self, indices_per_workers: List[List[List[Union[int, ChunkedIndex]]]]
    ) -> Iterator[List[Union[int, ChunkedIndex]]]:
        batches: List[List[Union[int, ChunkedIndex]]] = []
        counter = 0
        while sum([len(v) for v in indices_per_workers]) != 0:
            worker_indices = indices_per_workers[counter % self._num_workers]
            if len(worker_indices) == 0:
                batches.append([])
            else:
                batches.append(worker_indices.pop(0))
            counter += 1

        while True:
            if len(batches[-1]) == 0:
                batches.pop(-1)
            else:
                break

        yield from batches

    def _chunk_list(self, arr: List[Any], chunk_size: int) -> List[List[Any]]:
        out = []
        for i in range(0, len(arr), chunk_size):
            slice_item = slice(i, i + chunk_size, 1)
            out.append(arr[slice_item])
        return out
