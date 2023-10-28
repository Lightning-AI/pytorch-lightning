import math
import sys
from collections import Counter
from functools import partial
from typing import Any, Dict

import lightning
import pytest
import torch
from lightning.data.datasets.iterable import (
    DataLoader,
    LightningIterableDataset,
    _Chunk,
    _Stateful,
    _StatefulIterableDataset,
)


class Foo1:
    def state_dict(self, returned_samples: int) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class Foo2:
    def state_dict(self) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class Bar1:
    pass


class Bar2:
    def state_dict(self) -> Dict[str, Any]:
        pass


@pytest.mark.parametrize(
    ("klass", "fullfilled"),
    [
        pytest.param(Foo1, True),
        pytest.param(Foo2, True),
        pytest.param(Bar1, False),
        pytest.param(Bar2, False),
    ],
)
def test_serializable(klass, fullfilled):
    assert isinstance(klass(), _Stateful) == fullfilled


class DummyIterableDataset(_StatefulIterableDataset):
    def __init__(self, length: int):
        super().__init__()
        self.length = length
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        self.index += 1
        return 0


class WrongDummySerializableIterableDataset1(DummyIterableDataset):
    def state_dict(self):
        return {"length": self.length, "index": self.index}


class WrongDummySerializableIterableDataset2(DummyIterableDataset):
    def load_state_dict(self, state_dict):
        self.length = state_dict.pop("length")
        self.index = state_dict.pop("index")


class WorkingDummySerializableIterableDataset(
    WrongDummySerializableIterableDataset1, WrongDummySerializableIterableDataset2
):
    pass


@pytest.mark.parametrize(
    ("klass", "missing_method"),
    [
        pytest.param(WrongDummySerializableIterableDataset1, "load_state_dict"),
        pytest.param(WrongDummySerializableIterableDataset2, "state_dict"),
    ],
)
def test_required_abstract_methods_serializable_dataset(klass, missing_method):
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {klass.__name__} with abstract method.* {missing_method}",
    ):
        klass(10)


def test_serialization_iterable_dataset():
    dset = WorkingDummySerializableIterableDataset(10)

    dset_iter = iter(dset)

    assert dset_iter is dset

    for i in range(10):
        assert dset.state_dict() == {"length": 10, "index": i}
        next(dset_iter)
        assert dset.state_dict() == {"length": 10, "index": i + 1}


def test_iteration_serializable_iterable_dataset():
    dset = WorkingDummySerializableIterableDataset(10)

    i = 0

    for _ in dset:
        i = i + 1

    assert i == 10


def test_resume_iterable_dataset():
    dset1 = WorkingDummySerializableIterableDataset(10)
    dset1_iter = iter(dset1)

    for _ in range(5):
        next(dset1_iter)

    assert dset1.state_dict() == {"length": 10, "index": 5}

    dset2 = WorkingDummySerializableIterableDataset(12)
    dset2.load_state_dict(dset1.state_dict())

    assert dset2.length == 10
    assert dset2.index == 5

    i = 0
    for _ in dset2:
        i = i + 1

    assert i == 5

    dset2.length = 12
    for _ in dset2:
        i = i + 1

    assert i == 7
    assert dset2.state_dict() == {"length": 12, "index": 12}


class WrongChunkedDataset1(LightningIterableDataset):
    def load_chunk(self, curr_chunk: int):
        return [(curr_chunk, i) for i in range(self._chunk_size)]


class WrongChunkedDataset2(LightningIterableDataset):
    def load_sample_from_chunk(self, curr_chunk, curr_index):
        return curr_chunk[curr_index]


class WorkingChunkedDataset(WrongChunkedDataset1, WrongChunkedDataset2):
    pass


@pytest.mark.parametrize(
    ("klass", "missing_method"),
    [
        pytest.param(WrongChunkedDataset1, "load_sample_from_chunk"),
        pytest.param(WrongChunkedDataset2, "load_chunk"),
    ],
)
def test_required_abstract_methods_chunked_dataset(klass, missing_method):
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {klass.__name__} with abstract method.* {missing_method}",
    ):
        klass([10], 10)


def test_chunked_dataset_iteration():
    dset = WorkingChunkedDataset(list(range(5)), chunk_size=2, shuffle=False, wrap=False)

    curr_item = 0
    for i, item in enumerate(dset):
        assert item[0] == curr_item
        assert item[1] == i % 2
        curr_item += item[1]

    # goes to 4 but increases again in last item
    assert curr_item == 5
    assert i == 9


@pytest.mark.parametrize("lazy_shuffle", [False, True])
def test_chunk_dataset_iteration_shuffle(lazy_shuffle):
    dset = WorkingChunkedDataset(
        list(range(5)),
        chunk_size=2,
        shuffle=True,
        seed=12345,
        wrap=False,
        lazy_shuffle=lazy_shuffle,
    )
    counter = Counter()

    series = []
    unexpected_series = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    series_keys = []
    unexpected_series_keys = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    for item, key in dset:
        counter.update({item: 1})
        series.append(item)
        series_keys.append(key)

    for val in counter.values():
        assert val == 2

    # with shuffling it can't be equal to ordered!
    assert series != unexpected_series
    assert series_keys != unexpected_series_keys


def test_chunked_dataset_wrap():
    dset = WorkingChunkedDataset(list(range(5)), chunk_size=2, shuffle=True, seed=12345, wrap=True)

    dset_iter = iter(dset)

    # dataset has length 10, so this wraps 2 times
    for i in range(21):
        _ = next(dset_iter)


def test_chunked_dataset_resume_and_reset():
    dset = WorkingChunkedDataset(list(range(5)), chunk_size=2, shuffle=False, wrap=False)

    for i, item in enumerate(dset):
        assert item[0] == 0
        assert item[1] == i
        if i == 1:
            break

    # Every iterator starts from scratch
    for i, item in enumerate(dset):
        assert item[0] == 0
        assert item[1] == i
        if i == 1:
            break

    # this would be set when we load from state dict
    dset._start_index_sample = 1
    for i, item in enumerate(dset):
        assert item[0] == i
        assert item[1] == (i + 1) % 2
        if i == 1:
            break

    dset._start_index_chunk == 1
    for i, item in enumerate(dset):
        assert item[0] == 1
        assert item[1] == (i + 1) % 2
        if i == 1:
            break


@pytest.mark.parametrize("shuffle", [False, True])
def test_chunked_dataset_serialization(shuffle):
    dset = WorkingChunkedDataset(list(range(5)), chunk_size=2, shuffle=shuffle, wrap=False)

    assert dset.state_dict(0, 0) == {"current_chunk": 0, "current_sample_in_chunk": 0}

    dset_iter = iter(dset)
    assert dset.state_dict(0, 0) == {"current_chunk": 0, "current_sample_in_chunk": 0}

    dset.load_state_dict(dset.state_dict(0, 0))
    assert dset.state_dict(0, 0) == {"current_chunk": 0, "current_sample_in_chunk": 0}

    dset_iter = iter(dset)

    # throw away first few batches to alter internal state
    for i in range(3):
        next(dset_iter)

    curr_state = dset.state_dict(3, 0)

    original = [next(dset_iter) for _ in range(5)]

    dset.load_state_dict(curr_state)
    dset_iter = iter(dset)
    after_loading = [next(dset_iter) for _ in range(5)]

    # this isn't because we always skip to beginning of next chunk when loading and not already at beginning of chunk
    assert original != after_loading
    assert original[1:] == after_loading[:-1]

    # this actually puts us already on beginning of a chunk, but we'll forward to beginning of next chunk,
    # otherwise we'd two times resume from same checkpoint and assert equal behavior
    dset.load_state_dict(curr_state)
    _ = [next(dset_iter) for _ in range(2)]

    new_curr_state = dset.state_dict(6, 0)

    new_original = [next(dset_iter) for _ in range(3)]

    dset.load_state_dict(new_curr_state)
    new_after_loading = [next(dset_iter) for _ in range(3)]

    # this is equal since we exactly stopped at beginning of new chunk
    assert new_original == new_after_loading


class ChunkedTestDatasetDistributed(WorkingChunkedDataset):
    def _apply_sharding(self):
        super()._apply_sharding()

        assert len(self._local_chunks) == self.expected_num_chunks

        for i in range(1, len(self._local_chunks)):
            assert self._local_chunks[i]._chunk_data - self._local_chunks[i - 1]._chunk_data == self.expected_step_width


def sharding_test(fabric: lightning.Fabric, num_workers):
    dset = ChunkedTestDatasetDistributed(list(range(50)), 2, shuffle=False, wrap=False)

    num_shards = max(1, num_workers) * fabric.world_size

    # num_workers = 0 still has a single worker (the main process)
    expected_num_chunks = 50 // num_shards
    dset.expected_num_chunks = expected_num_chunks
    dset.expected_step_width = fabric.world_size * max(num_workers, 1)

    num_samples_per_rank = max(num_workers, 1) * 2 * expected_num_chunks
    loader = torch.utils.data.DataLoader(dset, num_workers=num_workers)

    for i, _ in enumerate(loader):
        fabric.barrier()

    assert i == num_samples_per_rank - 1


@pytest.mark.parametrize(
    ("num_workers", "world_size"),
    [
        pytest.param(0, 1),
        pytest.param(
            0,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            1,
            1,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            1,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            2,
            1,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            2,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
    ],
)
def test_sharding(num_workers, world_size):
    fabric = lightning.Fabric(accelerator="cpu", devices=world_size, strategy="ddp_spawn")
    fabric.launch(partial(sharding_test, num_workers=num_workers))


def sharding_resume_test(fabric: lightning.Fabric, num_workers):
    chunk_size = 2
    dset = WorkingChunkedDataset(list(range(100)), chunk_size, shuffle=False, wrap=False)
    loader = torch.utils.data.DataLoader(dset, num_workers=num_workers, shuffle=False)
    num_shards = max(1, num_workers) * fabric.world_size

    for i in [23, 37, 10, 20]:
        curr_index = math.ceil(i / num_shards / chunk_size) * num_shards * chunk_size
        next_chunk = math.ceil(curr_index / chunk_size)

        curr_state = dset.state_dict(i, num_workers=num_workers)
        assert curr_state == {"current_chunk": next_chunk, "current_sample_in_chunk": 0}

        dset.load_state_dict(curr_state)
        loader = torch.utils.data.DataLoader(dset, num_workers=num_workers, shuffle=False)

        # calculate starting chunks
        # next_chunk + fabric.global_rank * max(1,num_workers) determines the base offset for each rank
        # i % chunk_size makes sure that workers are alternating
        #   e.g. w0 returns first element of first chunk then w1 returns first element of second chunk then w0 returns
        #   second element of first chunk etc.
        # i // num_shards * num_shards progresses to next chunks
        curr_worker_chunk = {
            i: next_chunk
            + fabric.global_rank * max(1, num_workers)
            + i % max(1, num_workers)
            + i // (chunk_size * num_shards)
            for i in range(max(1, num_workers))
        }
        curr_worker_chunk_elem = {i: 0 for i in range(max(1, num_workers))}

        for i, batch in enumerate(loader):
            curr_worker = i % max(1, num_workers)
            assert batch[0] == curr_worker_chunk[curr_worker]
            assert batch[1] == curr_worker_chunk_elem[curr_worker]

            curr_worker_chunk_elem[curr_worker] += 1

            if curr_worker_chunk_elem[curr_worker] == chunk_size:
                curr_worker_chunk[curr_worker] += num_shards
                curr_worker_chunk_elem[curr_worker] = 0
        fabric.barrier()


@pytest.mark.parametrize(
    ("num_workers", "world_size"),
    [
        pytest.param(0, 1),
        pytest.param(
            0,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            1,
            1,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            1,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            2,
            1,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            2,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
    ],
)
def test_chunked_dataset_sharded_state_dict_resume(num_workers, world_size):
    fabric = lightning.Fabric(accelerator="cpu", devices=world_size, strategy="ddp_spawn")
    fabric.launch(partial(sharding_resume_test, num_workers=num_workers))


@pytest.mark.parametrize("chunk_size", [20, 30, 40])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("shuffle_seed", [None, 123])
@pytest.mark.parametrize("delayed_start", [False, True])
def test_chunk(chunk_size, shuffle, shuffle_seed, delayed_start):
    data = list(range(chunk_size))
    delayed_start_index = int(delayed_start) * (chunk_size - 10)
    chunk = _Chunk(data, chunk_size=chunk_size, start_index=delayed_start_index)
    linear_permutation = tuple(range(chunk_size))
    assert chunk.index_permutations == linear_permutation

    for i, index in enumerate(chunk):
        assert index == delayed_start_index + i

    assert chunk.chunk_size == chunk_size

    if shuffle:
        generator = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed else None

        chunk = chunk.shuffle(generator=generator)

        old_permutation = chunk.index_permutations
        assert old_permutation != linear_permutation

        new_perm = []

        for i, index in enumerate(chunk):
            new_perm.append(index)

        assert tuple(new_perm) == tuple([old_permutation[k] for k in range(delayed_start_index, chunk_size)])
        assert len(new_perm) == chunk_size - delayed_start_index

        if shuffle_seed:
            chunk = chunk.shuffle(generator=generator.manual_seed(shuffle_seed))
            assert chunk.index_permutations == old_permutation

        assert chunk.chunk_size == chunk_size


class MyDataset(_StatefulIterableDataset):
    def __init__(self, length):
        self.length = length
        self.samples = list(range(length))
        self.curr_iter = 0

    def __iter__(self):
        for sample in self.samples[self.curr_iter :]:
            yield sample
            self.curr_iter += 1

    def state_dict(self, returned_samples, num_workers):
        return {"curr_iter": returned_samples, "num_workers": num_workers}

    def load_state_dict(self, state_dict):
        self.curr_iter = state_dict.pop("curr_iter")


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize(
    "num_workers",
    [
        pytest.param(0),
        pytest.param(
            1,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
    ],
)
@pytest.mark.parametrize("prefetch_factor", [1, 2, 3])
@pytest.mark.parametrize("length", [100, 101])
@pytest.mark.parametrize("num_batches", [1, 2, 7])
def test_resumable_loader(batch_size, num_workers, prefetch_factor, length, num_batches):
    dset = MyDataset(length)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    loader_iter = iter(loader)
    for i, batch in enumerate(loader_iter):
        assert loader._get_batch_size(batch) == batch_size
        if i == num_batches - 1:
            break

    state_dict = loader.state_dict()
    assert state_dict["returned_samples"] == batch_size * num_batches
    assert state_dict["dataset"] == {
        "curr_iter": batch_size * num_batches,
        "num_workers": num_workers,
    }

    state_dict["returned_samples"] += 1
    state_dict["dataset"]["curr_iter"] += 1
    loader.load_state_dict(state_dict)
    assert loader.returned_samples == batch_size * num_batches + 1
    assert loader.dataset.curr_iter == batch_size * num_batches + 1


def test_state_dict_error():
    loader = DataLoader([1, 2, 3])
    with pytest.raises(
        TypeError,
        match="The dataset has no method `state_dict` that accepts `returned_samples` and `num_workers`",
    ):
        loader.state_dict()


def test_load_state_dict_error():
    loader = DataLoader([1, 2, 3])
    with pytest.raises(
        TypeError,
        match="The dataset has no method `load_state_dict` accepting a `state_dict`",
    ):
        loader.load_state_dict({"returned_samples": 1, "dataset": {"some_key": "some_val"}})
