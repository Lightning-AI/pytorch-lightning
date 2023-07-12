import math
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generator, List, Literal, Optional, Protocol, runtime_checkable, Sequence, Tuple

import torch
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import IterableDataset

from lightning.data.datasets.base import _Dataset
from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv, Environment


class _StatefulIterableDataset(ABC, IterableDataset):
    @abstractmethod
    def state_dict(self, returned_samples: int, num_workers: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class _Chunk:
    """A single chunk of data.

    Args:
        chunk_data: The original data contained by this chunk
        chunk_size: The number of samples contained in this chunk
        start_index: the index from where to start sampling the chunk (already retrieved samples)

    """

    def __init__(self, chunk_data: Any, chunk_size: int, start_index: int = 0):
        self._chunk_data = chunk_data
        self._index_permutations: Optional[Tuple[int, ...]] = None
        self._start_index = start_index
        self._chunk_size = chunk_size

    def shuffle(self, generator: Optional[torch.Generator] = None) -> "_Chunk":
        """Shuffles index permutations for the current chunk."""
        new_indices = torch.randperm(self.chunk_size, generator=generator, device="cpu").tolist()
        self._index_permutations = tuple(new_indices)
        return self

    def __iter__(self) -> Generator[int, None, None]:
        """Returns an iterator over the index permutations."""
        # iterates over indices
        index_permutations = self.index_permutations
        for i in range(self._start_index, self.chunk_size):
            yield index_permutations[i]

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def index_permutations(self) -> Tuple[int, ...]:
        if self._index_permutations is None:
            return tuple(range(self._chunk_size))
        return self._index_permutations


class LightningIterableDataset(_StatefulIterableDataset, _Dataset):
    """An iterable dataset that can be resumed mid-epoch, implements chunking and sharding of chunks. The behavior of
    this dataset can be customized with the following hooks:

    - ``prepare_chunk`` gives the possibility to prepare the chunk one iteration before its actually loaded
        (e.g. download from s3).
    - ``load_chunk`` implements how an entire chunk is loaded into memory
        (e.g. loading the previously downloaded file into memory)
    - ``load_sample_from_chunk`` implements how to retrieve a single sample from the current chunk
        (e.g. indexing the chunk if it's a list or just returning it if the chunk has a size of 1)

    Args:
        chunks: The chunked_data to load.
        chunk_size: The number of samples in each chunk
        num_parallel_chunks: How many chunks to load in parallel.
        env: The compute-environment. Important for sharding. Contains the distributed world-size,
            the distributed global rank, the number of workers on the current rank and the current worker rank.
            If None, it will try to detect these things automatically.
        shuffle: Whether to shuffle your data. Will shuffle both, order of chunks before sharding and order of
            samples within each chunk.
        seed: The seed for the random number generator. If :param:`shuffle` = False, the seed has no effect.
        wrap: Whether to restart your dataset if it's exhausted. If set to True, it results in a
            virtually infinite dataset looping through the same data over and over again.
        lazy_shuffle: Whether to shuffle your data lazily instead of upfront.
            This consumes a lot less memory, but may yield undeterministic results.
        backend: A string pointing to the respective cloud-backend to use. Currently "s3" and "local" are supported.

        Note:
            :param:`lazy_shuffle` is experimental, consumes less memory than shuffling everything in advance (default)
            but may result in undeterministic behavior.

        Note:
            On resume from a state-dict, we always skip currently started chunks as these would make the data-order
            impossible to determine with sharding. Upon resuming from a point, where a new chunk would be started
            anyways, nothing is skipped.

        Note:
            Order of data is only guaranteed when resuming with the same distributed settings and the same number of
            workers. Everything else leads to different sharding and therefore results in different data order.

    """

    def __init__(
        self,
        chunks: Sequence[Any],
        chunk_size: int = 1,
        num_parallel_chunks: int = 1,
        env: Optional[Environment] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        wrap: bool = False,
        lazy_shuffle: bool = False,
        backend: Literal["local", "s3"] = "local",
    ):
        _StatefulIterableDataset.__init__(self)
        _Dataset.__init__(self, backend=backend)

        chunks = [_Chunk(c, chunk_size=chunk_size) for c in chunks]
        if env is None:
            # must detect distributed env here since distributed is not initialized in worker processes with ddp spawn
            # can't detect worker env here since workers not yet initialized
            env = Environment(_DistributedEnv.detect(), None)
        self._env = env
        self._shuffle = shuffle
        self._lazy_shuffle = lazy_shuffle

        # prepare shuffling
        if shuffle:
            generator = torch.Generator()
            if seed is not None:
                generator = generator.manual_seed(seed)
        else:
            generator = None

        self._seed = seed
        self._generator = generator
        self._initial_generator_state = self._generator.get_state() if self._generator is not None else None

        self._num_parallel_chunks = num_parallel_chunks
        self._chunks = chunks
        self._original_chunks = chunks

        self._chunk_size = chunk_size
        self._local_chunks: List[_Chunk] = []
        self._wrap = wrap

        self._start_index_chunk = 0
        self._start_index_sample = 0
        self._curr_chunk_index = 0
        self._curr_sample_index = 0

        self._curr_loaded_chunks: List[_Chunk] = []
        self._curr_loaded_num_samples = 0

    @abstractmethod
    def load_chunk(self, chunk: Any) -> Any:
        """Implement this to load a single chunk into memory. This could e.g. mean loading the file that has previously
        been downloaded from s3.

        Args:
            chunk: The chunk that should be currently loaded

        """

    @abstractmethod
    def load_sample_from_chunk(self, chunk: Any, index: int) -> Any:
        """Implement this to retrieve a single sample from a given (already loaded) chunk. This could be indexing a
        list or returning the entire chunk if it's size is 1.

        Args:
            chunk: The chunk the sample should be retrieved from
            index: The index of the current sample to retrieve within the chunk.

        """

    def prepare_chunk(self, chunk: Any) -> None:
        """Prepares a single chunk before it is actually loaded. This could e.g. download the actual file from s3.

        Args:
            chunk: the chunk data to prepare.

        """

    def __iter__(self) -> "LightningIterableDataset":
        """Creates an iterator.

        Before that, detects the env if necessary, shuffles chunks, shards the data and shuffles sample orders within
        chunks.

        """
        self._curr_chunk_index = self._start_index_chunk
        self._curr_sample_index = self._start_index_sample
        if self._env.worker_env is None:
            self._env.worker_env = _WorkerEnv.detect()

        self._chunks = self._shuffle_if_necessary(self._chunks, 0, shuffle_chunk_order=True, shuffle_sample_order=False)
        self._apply_sharding()
        self._local_chunks = self._shuffle_if_necessary(
            self._local_chunks,
            self._curr_chunk_index,
            shuffle_chunk_order=False,
            shuffle_sample_order=True,
        )
        self._ensure_chunks_loaded()
        return self

    def __next__(self) -> Any:
        """Returns the next sample.

        If necessary, this also loads the new chunks.

        """
        self._check_if_sharded()
        self._ensure_chunks_loaded()

        if self._curr_sample_index >= self._curr_loaded_num_samples:
            self._curr_chunk_index += self._num_parallel_chunks
            self._check_dataset_end()

            self._load_next_chunks()
            self._curr_sample_index = 0

        remainder = self._curr_sample_index
        curr_loaded_chunk_idx = 0
        for i, c in enumerate(self._curr_loaded_chunks):
            if c.chunk_size > remainder:
                curr_loaded_chunk_idx = i
                break

            remainder -= c.chunk_size

        sample = self.load_sample_from_chunk(
            self._curr_loaded_chunks[curr_loaded_chunk_idx]._chunk_data,
            self._curr_loaded_chunks[curr_loaded_chunk_idx].index_permutations[remainder],
        )
        self._curr_sample_index += 1

        return sample

    def state_dict(self, returned_samples: int, num_workers: int) -> Dict[str, Any]:
        """Returns a global state-dict across all shards and workers. For construction of a global state-dict the
        `returned_samples` and `num_workers` arguments are required, since the main process, which is taking this
        state-dict, typically does not have access to worker_info.

        Args:
            returned_samples: the number of totally returned samples by the dataloader(s) (across all distributed
                training processes).
            num_workers: number of dataloader workers per distributed training process.

        """

        # compute indices locally again since other workers may have different offsets
        if num_workers == 0:
            # num_workers=0 indicate loading in the main process --> main process becomes 1 effective worker
            num_workers = 1

        # manually compute num_shards since env doesn't know about num_workers in main process outside dataloader iter
        assert self._env.dist_env is not None
        num_shards = self._env.dist_env.world_size * num_workers

        # fast-forward so that each chunk on each shard is finished -> this may skip a few samples!
        curr_index = math.ceil(returned_samples / num_shards / self._chunk_size) * num_shards * self._chunk_size

        # since we go to next chunk, always start at beginning of chunk
        curr_sample_in_chunk = 0

        # global chunk index
        curr_chunk_index = math.ceil(curr_index / self._chunk_size)
        return {
            "current_chunk": curr_chunk_index,
            "current_sample_in_chunk": curr_sample_in_chunk,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads a previous state dict to resume for it's state.

        Args:
            state_dict: the previous state-dict containing internal indices and random number generator states.

        Note:
            Some of the changes only take effect when creating a new iterator

        """
        state_dict = deepcopy(state_dict)
        self._start_index_chunk = state_dict.pop("current_chunk")
        self._start_index_sample = state_dict.pop("current_sample_in_chunk")

        self._curr_chunk_index = self._start_index_chunk
        self._curr_sample_index = self._start_index_sample

        self._curr_loaded_chunks = []
        self._curr_loaded_num_samples = 0

    def _ensure_chunks_loaded(self) -> None:
        """Ensures that the correct number of chunks is loaded."""
        if len(self._curr_loaded_chunks) != self._num_parallel_chunks:
            self._check_dataset_end()
            self._load_next_chunks()

    def _load_next_chunks(self) -> None:
        """Loads the current chunks and prepares the chunks thereafter."""
        self._curr_loaded_chunks = []
        self._curr_loaded_num_samples = 0
        # load next N chunks
        for i in range(self._num_parallel_chunks):
            curr_chunk = self._local_chunks[self._curr_chunk_index + i]
            loaded_chunk = _Chunk(
                self.load_chunk(curr_chunk._chunk_data),
                chunk_size=curr_chunk.chunk_size,
                start_index=curr_chunk._start_index,
            )
            if self._lazy_shuffle:
                loaded_chunk.shuffle(generator=self._generator)
            else:
                loaded_chunk._index_permutations = curr_chunk.index_permutations
            self._curr_loaded_chunks.append(loaded_chunk)
            self._curr_loaded_num_samples += loaded_chunk.chunk_size

        # prepare the next N chunks after currently loaded ones
        for i in range(self._num_parallel_chunks, self._num_parallel_chunks * 2):
            if self._curr_chunk_index + i >= len(self._local_chunks):
                break
            curr_chunk = self._local_chunks[self._curr_chunk_index + i]
            self.prepare_chunk(curr_chunk._chunk_data)

    def _apply_sharding(self) -> None:
        """Shards the chunks if necessary.

        No-op if already sharded

        """
        if not self._local_chunks:
            num_shards = self._env.num_shards

            # every shard must have the same number of chunks -> truncate if not evenly divisible
            max_chunks = len(self._chunks) // num_shards * num_shards
            self._local_chunks = self._chunks[self._env.shard_rank : max_chunks : num_shards]

        # if state-dict was set, the curr chunk index was the global number across all shards
        # --> divide it to get local number
        if self._start_index_chunk and self._start_index_chunk == self._curr_chunk_index:
            self._curr_chunk_index = math.ceil(self._curr_chunk_index // self._env.num_shards)

    def _check_if_sharded(self) -> None:
        """Raises a warning if the dataset is not sharded."""
        if not self._local_chunks:
            warnings.warn(
                "Chunks have not been sharded yet. Call iter() on your dataset to ensure sharding is done correctly. "
                "It won't recognize dataloader workers when manually calling it outside an actual dataloader."
            )

    def _check_dataset_end(self) -> None:
        """Checks if the dataset has reached it's end or should be restarted."""
        if self._curr_chunk_index >= len(self._local_chunks):
            if self._wrap:
                self._curr_chunk_index = 0
            else:
                raise StopIteration

    def _shuffle_if_necessary(
        self,
        chunks: List[_Chunk],
        first_chunk_index: int,
        shuffle_chunk_order: bool = True,
        shuffle_sample_order: bool = True,
    ) -> List[_Chunk]:
        """This shuffles the chunk-order and the order of samples within each chunk.

        Args:
            chunks: The chunks to optionally shuffle
            first_chunk_index: The point to which the generator should be replayed
            shuffle_chunk_order: Whether to shuffle the order of chunks
            shuffle_sample_order: Whether to shuffle the order of samples within a chunk

        """
        # re-seed generator
        if self._generator is not None and self._initial_generator_state is not None:
            self._generator = self._generator.set_state(self._initial_generator_state)

        # shuffle chunks if necessary
        chunks = _Chunk(chunks, len(chunks))
        if self._shuffle and shuffle_chunk_order:
            chunks.shuffle(generator=self._generator)
        # this is annoying but otherwise we cannot make sure the states are the same
        elif self._shuffle:
            _dummy_chunks = _Chunk(None, len(chunks._chunk_data))
            _dummy_chunks.shuffle(generator=self._generator)
        chunks = [chunks._chunk_data[i] for i in chunks]

        if not shuffle_sample_order:
            return chunks

        # after shuffling all chunks -> fast forward to first_chunk_index
        if self._shuffle and self._lazy_shuffle:
            for _ in range(first_chunk_index):
                chunk = _Chunk(None, chunk_size=self._chunk_size)
                chunk.shuffle(generator=self._generator)
        # shuffle samples within each chunk
        elif self._shuffle:
            chunks = [c.shuffle(generator=self._generator) for c in chunks]

        return chunks


class DataLoader(_DataLoader):
    __doc__ = _DataLoader.__doc__

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.returned_samples = 0

    def __iter__(self) -> Generator[Any, None, None]:  # type: ignore
        base_iter = super().__iter__()

        for batch in base_iter:
            self.returned_samples += self._get_batch_size(batch)
            yield batch

    def _get_batch_size(self, batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        if isinstance(batch, Sequence):
            return len(batch[0])

        assert isinstance(self.batch_size, int)
        return self.batch_size

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state-dict of the dataset."""
        if isinstance(self.dataset, _Stateful):
            state_dict = self.dataset.state_dict(returned_samples=self.returned_samples, num_workers=self.num_workers)
            return {"returned_samples": self.returned_samples, "dataset": state_dict}

        raise TypeError("The dataset has no method `state_dict` that accepts `returned_samples` and `num_workers`")

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads a given state-dict onto the dataset."""
        self.returned_samples = state_dict.pop("returned_samples")
        if isinstance(self.dataset, _Stateful):
            return self.dataset.load_state_dict(state_dict["dataset"])

        raise TypeError("The dataset has no method `load_state_dict` accepting a `state_dict`")


@runtime_checkable
class _Stateful(Protocol):
    def state_dict(self, returned_samples: int, num_workers: int) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
