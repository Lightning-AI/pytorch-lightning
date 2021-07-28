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
import math
import os
import random
from collections.abc import Iterable
from typing import Optional
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.data._utils.worker import get_worker_info
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.dataset import Dataset, IterableDataset

import tests.helpers.utils as tutils
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import (
    _dataloader_load_state_dict,
    _dataloader_to_state_dict,
    CaptureIterableDataset,
    FastForwardSampler,
)
from pytorch_lightning.utilities.enums import AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.runif import RunIf


# Credit to PyTorch Team.
# Taken from:
# https://github.com/pytorch/pytorch/blob/3b977a0d2834d300c0301a0c6af98c8e939019ce/torch/utils/data/_utils/worker.py#L151
# Not available until torch 1.9.0
def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def test_fast_forward_getattr():
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)

    assert index_batch_sampler.batch_size == 3
    assert index_batch_sampler.sampler == sampler


def test_fast_forward_on_batch_sampler():
    """
    This test ensures ``FastForwardSampler`` applied to ``BatchSampler`` correctly retrived
    the right next batch on restart.
    """
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)

    assert isinstance(index_batch_sampler, Iterable)

    index_batch_sampler_iter = iter(index_batch_sampler)

    assert next(index_batch_sampler_iter) == [0, 1, 2]
    assert next(index_batch_sampler_iter) == [3, 4, 5]

    state_dict = index_batch_sampler.state_dict(2)

    index_batch_sampler = FastForwardSampler(batch_sampler)
    index_batch_sampler.load_state_dict(state_dict)

    index_batch_sampler_iter = iter(index_batch_sampler)
    assert next(index_batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_sequential_sampler():
    """
    This test ensures ``FastForwardSampler`` applied to ``SequentialSampler`` correctly retrived
    the right next batch on restart.
    """
    dataset = range(15)
    sequential_sampler = SequentialSampler(dataset)
    sampler = FastForwardSampler(sequential_sampler)
    sampler.setup(3)
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [0, 1, 2]
    assert next(batch_sampler_iter) == [3, 4, 5]

    state_dict = sampler.state_dict(2)
    assert state_dict[0]["current_iteration"] == 6

    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == [6, 7, 8]


@pytest.mark.skipif(torch.cuda.is_available(), reason="todo (tchaton) Need more investigation")
def test_fast_forward_on_random_sampler():
    """
    This test ensures ``FastForwardSampler`` applied to ``RandomSampler`` correctly retrived
    the right next batch on restart.
    """
    seed = 42
    seed_everything(42)

    dataset = range(15)
    generator = torch.Generator().manual_seed(seed)
    values = list(RandomSampler(dataset, generator=generator))

    generator = torch.Generator().manual_seed(seed)
    random_sampler = RandomSampler(dataset, generator=generator)
    sampler = FastForwardSampler(random_sampler)
    sampler.setup(3)
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == values[:3]
    assert next(batch_sampler_iter) == values[3:6]
    assert next(batch_sampler_iter) == values[6:9]

    state_dict = sampler.state_dict(3)
    assert state_dict[0]["current_iteration"] == 9
    state_dict[0]["current_iteration"] = 6

    seed_everything(42)
    generator = torch.Generator().manual_seed(seed)
    random_sampler = RandomSampler(dataset, generator=generator)
    sampler = FastForwardSampler(random_sampler)
    sampler.setup(3)
    batch_sampler = BatchSampler(sampler, 3, False)
    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == values[6:9]
    has_raised = False
    try:
        for _ in range(5):
            next(batch_sampler_iter)
    except StopIteration:
        has_raised = True
        assert sampler._current_iteration == 0
        sampler.load_state_dict(sampler.state_dict(0))
    assert has_raised


class RangeIterableDataset(IterableDataset):
    def __init__(self, data, num_workers: int, batch_size: int, state_dict=None, attr_name: str = "iter_sampler"):
        self.data = list(data)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state_dict = state_dict
        self.attr_name = attr_name

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info and self.num_workers == 2:
            id = worker_info.id
            num_samples = len(self.data)
            if id == 0:
                self.data = list(self.data)[: num_samples // 2]
            else:
                self.data = list(self.data)[num_samples // 2 :]
            self.user_sampler = RandomSampler(self.data)
        else:
            self.user_sampler = RandomSampler(self.data)

        setattr(self, self.attr_name, iter(self.user_sampler))
        return self

    def __next__(self):
        iter_sampler = getattr(self, self.attr_name)
        return self.data[next(iter_sampler)]


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test takes around 30 sec and should be skipped in Azure CI")
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_fast_forward_sampler_over_iterative_dataset(num_workers):
    """
    This test ensures ``FastForwardSampler`` and ``CaptureIterableDataset`` are properly being
    used to capture workers states.
    """
    batch_size = 3
    initial_seed = seed_everything(42)
    generator = torch.Generator()
    generator.manual_seed(initial_seed)
    dataset = RangeIterableDataset(range(20), num_workers, batch_size, True)
    dataset = CaptureIterableDataset(dataset, num_workers)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    Trainer._add_sampler_metadata_collate(dataloader)

    iter_dataloader = iter(dataloader)
    batches = []
    for _ in range(5):
        batches.append(next(iter_dataloader))

    # restarting on batch_1 and getting 3 extra batches

    state_dict = {"iter_sampler": {}}
    for batch in batches[:2]:
        batch, _state_dict = CaptureIterableDataset.extract_samplers_state_dict_from_batch(batch)
        for k, v in _state_dict[0].items():
            state_dict[k].update(v)

    assert len(state_dict["iter_sampler"]) == (num_workers if num_workers > 1 else 1)

    initial_seed = seed_everything(42)
    generator.manual_seed(initial_seed)
    dataset = RangeIterableDataset(range(20), num_workers, batch_size, state_dict=state_dict)
    dataset = CaptureIterableDataset(dataset)
    dataset.load_state_dict(state_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    Trainer._add_sampler_metadata_collate(dataloader)

    iter_dataloader = iter(dataloader)
    batches_restart = []
    for _ in range(3):
        batches_restart.append(next(iter_dataloader))

    assert torch.equal(batches_restart[0]["data"], batches[2]["data"])
    assert torch.equal(batches_restart[1]["data"], batches[3]["data"])
    assert torch.equal(batches_restart[2]["data"], batches[4]["data"])


def _setup_ddp(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _test_fast_forward_sampler_with_distributed_sampler(rank, worldsize):
    _setup_ddp(rank, worldsize)

    initial_seed = seed_everything(42)

    generator = torch.Generator()
    generator.manual_seed(initial_seed)

    num_workers = 2
    batch_size = 4

    dataset = range(30)
    sampler = FastForwardSampler(DistributedSampler(dataset, num_replicas=worldsize, rank=rank, seed=initial_seed))
    sampler.setup(batch_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, generator=generator, sampler=sampler
    )

    iter_dataloader = iter(dataloader)

    num_yielded = 0
    batches = []
    while True:
        try:
            batches.append(next(iter_dataloader))
            num_yielded += 1
        except StopIteration:
            break

    expected = torch.tensor([17, 27, 24]) if rank == 0 else torch.tensor([19, 5, 3])
    assert torch.equal(batches[-1], expected)

    assert sampler.state_dict(num_yielded)[0]["current_iteration"] == 16

    reload_state_dict = sampler.state_dict(num_yielded - 1)
    assert reload_state_dict[0]["current_iteration"] == 12

    sampler = FastForwardSampler(DistributedSampler(dataset, num_replicas=worldsize, rank=rank, seed=initial_seed))
    sampler.setup(batch_size)
    sampler.load_state_dict(reload_state_dict)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, generator=generator, sampler=sampler
    )

    iter_dataloader = iter(dataloader)

    batches = []
    while True:
        try:
            batches.append(next(iter_dataloader))
        except StopIteration:
            break

    assert torch.equal(batches[-1], expected)
    assert sampler.state_dict(num_yielded)[0]["current_iteration"] == 16


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test takes around 25 sec and should be skipped in Azure CI")
@RunIf(skip_windows=True)
def test_fast_forward_sampler_with_distributed_sampler():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(_test_fast_forward_sampler_with_distributed_sampler, args=(worldsize,), nprocs=worldsize)


class MetaLearningDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        task_num_classes: int = 5,
        num_workers: Optional[int] = None,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        initial_seed: Optional[int] = None,
        shuffle: bool = True,
        debugging: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers or 1
        self.global_rank = global_rank
        self.world_size = world_size
        self.task_num_classes = task_num_classes
        self.labels = labels = getattr(dataset, "labels")
        self.initial_seed = initial_seed
        self.generator: Optional[torch.Generator] = None
        self.current_task_iteration = 0
        self.shuffle = shuffle
        self.debugging = debugging

        if labels is None:
            raise MisconfigurationException(f"Provided {self.dataset} should have an attribute labels.")

        if len(labels) != len(dataset):
            raise MisconfigurationException("Found provided ``labels`` don't match the dataset length.")

        if (isinstance(global_rank, int) and world_size is None) or (
            isinstance(world_size, int) and global_rank is None
        ):  # noqa E129
            raise MisconfigurationException("Both ``world_size`` and ``global_rank`` should be provided !")

        self.unique_labels = np.unique(self.labels)

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size is not None and self.world_size > 1

    def set_seed(self, shared: bool = False):
        initial_seed = self.initial_seed + self.current_task_iteration
        if shared:
            seed = initial_seed
            np_seed = _generate_state(initial_seed, 0)
        else:
            seed = initial_seed + self.worker_id + self.global_rank + self.current_task_iteration
            np_seed = _generate_state(initial_seed, self.worker_id + self.global_rank)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(np_seed)

    def sample_task_indices(self):
        self.set_seed(shared=True)
        self.selected_indexes = np.random.choice(self.unique_labels, self.task_num_classes, replace=False)
        self.selected_indexes.sort()

        # subset of indices from the entire dataset where the labels are actually among the
        # task_num_classes selected_indexes

        self.task_indices = np.arange(len(self.dataset))[np.in1d(self.labels, self.selected_indexes)]
        self.task_length = len(self.task_indices)
        self.set_seed(shared=False)

    @property
    def worker_rank(self) -> int:
        worker_id = self.worker_id
        is_global_zero = self.global_rank == 0
        return self.global_rank + worker_id + int(not is_global_zero)

    def create_sampler(self):
        data = range(self.task_length)
        if self.world_size == 1 and self.num_workers in (0, 1):
            if self.shuffle:
                self.sampler = RandomSampler(data, generator=self.generator)
            else:
                self.sampler = SequentialSampler(data)
        else:
            num_workers = 1 if self.num_workers in (None, 0) else self.num_workers
            num_replicas = num_workers * self.world_size
            current_seed = self.initial_seed + self.current_task_iteration
            self.sampler = DistributedSampler(
                data, num_replicas=num_replicas, rank=self.worker_rank, shuffle=self.shuffle, seed=current_seed
            )

    def __iter__(self):
        if self.generator is None:
            self.generator = torch.Generator().manual_seed(self.initial_seed)
        self.sample_task_indices()
        self.create_sampler()
        self.batch_sampler = BatchSampler(self.sampler, batch_size=self.batch_size, drop_last=self.drop_last)
        self.iter_sampler = iter(self.batch_sampler)
        self.is_first_batch = True
        self.current_task_iteration += 1
        return self

    def increment_iteration(self):
        self.current_task_iteration += 1

    def __next__(self):
        # this is optional, but useful to accumulate gradient over the entire task.
        is_first_batch = self.is_first_batch if self.debugging else (self.is_first_batch and self.worker_id == 0)
        if is_first_batch:
            self.is_first_batch = False
            return {"task_length": len(self.batch_sampler), "selected_indexes": self.selected_indexes}

        random_indices = next(self.iter_sampler)
        task_indices = [self.task_indices[idx] for idx in random_indices]
        return default_collate([self.dataset[idx] for idx in task_indices])


class ClassificationDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)

    def __getitem__(self, index):
        return (self.inputs[index], self.labels[index])

    def __len__(self):
        return len(self.inputs)


def _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset(rank, worldsize):
    if worldsize > 1:
        _setup_ddp(rank, worldsize)

    def all_gather(tensor, world_size):
        tensor_list = [torch.zeros_like(tensor, dtype=torch.int64) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, tensor)
        return tensor_list

    initial_seed = seed_everything(42)

    generator = torch.Generator()
    generator.manual_seed(initial_seed)

    num_workers = 2
    batch_size = 4
    dataset_length = 60
    num_classes = 10

    labels = np.random.randint(0, num_classes, dataset_length)

    dataset = ClassificationDataset(range(dataset_length), labels)
    dataset = MetaLearningDataset(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        global_rank=rank,
        world_size=worldsize,
        initial_seed=initial_seed,
        debugging=True,
        shuffle=True,
    )
    dataset = CaptureIterableDataset(dataset, initial_seed=initial_seed)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, generator=generator)
    Trainer._add_sampler_metadata_collate(dataloader)

    epoch_results = []
    for _ in range(2):
        iter_dataloader = iter(dataloader)
        batches = []
        while True:
            try:
                batches.append(next(iter_dataloader))
            except StopIteration:
                break
        epoch_results.append(batches)
        dataloader.dataset.dataset.current_task_iteration += 1

    assert len(epoch_results) == 2

    assert len(epoch_results[0]) == math.ceil((dataset_length / (num_workers * worldsize)) / batch_size) + 2

    if worldsize == 1:
        assert epoch_results[0][0]["data"]["task_length"] == epoch_results[0][1]["data"]["task_length"]
        assert torch.equal(
            epoch_results[0][0]["data"]["selected_indexes"], epoch_results[0][1]["data"]["selected_indexes"]
        )
        assert 0 in epoch_results[0][2][AutoRestartBatchKeys.PL_SAMPLERS]["iter_sampler"]  # worker id 0
        assert 1 in epoch_results[0][3][AutoRestartBatchKeys.PL_SAMPLERS]["iter_sampler"]  # worker id 1
        assert not torch.equal(epoch_results[0][2]["data"][0], epoch_results[0][3]["data"][0])
    else:
        first_task_metadata = all_gather(epoch_results[0][0]["data"]["task_length"], worldsize)
        second_task_metadata = all_gather(epoch_results[0][1]["data"]["task_length"], worldsize)
        assert torch.equal(first_task_metadata[0], first_task_metadata[1])
        assert torch.equal(second_task_metadata[0], second_task_metadata[1])
        assert torch.equal(first_task_metadata[0], second_task_metadata[1])

        first_batch_list = all_gather(epoch_results[0][2]["data"][0], worldsize)
        assert not torch.equal(first_batch_list[0], first_batch_list[1])
        second_batch_list = all_gather(epoch_results[0][3]["data"][0], worldsize)
        assert not torch.equal(second_batch_list[0], second_batch_list[1])

    # restarting on epoch 0 / real batch 2
    state_dict = {"iter_sampler": {}}
    for batch in epoch_results[0][2:4]:
        batch, _state_dict = CaptureIterableDataset.extract_samplers_state_dict_from_batch(batch)
        for k, v in _state_dict[0].items():
            state_dict[k].update(v)

    dataset = ClassificationDataset(range(dataset_length), labels)
    dataset = MetaLearningDataset(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        global_rank=rank,
        world_size=worldsize,
        initial_seed=initial_seed,
        debugging=True,
        shuffle=True,
    )

    dataset = CaptureIterableDataset(dataset, initial_seed=initial_seed)
    dataset.load_state_dict(state_dict)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, generator=generator)
    Trainer._add_sampler_metadata_collate(dataloader)

    epoch_results_restart = []
    for _ in range(2):
        iter_dataloader = iter(dataloader)
        batches = []
        while True:
            try:
                batches.append(next(iter_dataloader))
            except StopIteration:
                break
        epoch_results_restart.append(batches)
        dataloader.dataset.dataset.increment_iteration()
        dataloader.dataset.reset_on_epoch()

    assert len(epoch_results_restart[0]) + 2 == len(epoch_results[0])
    epoch_tensors = [e["data"][0] for e in epoch_results[0][4:]]
    epoch_tensors_restart = [e["data"][0] for e in epoch_results_restart[0][2:]]

    for t, tr in zip(epoch_tensors, epoch_tensors_restart):
        assert torch.equal(t, tr)

    epoch_tensors = [e["data"][0] for e in epoch_results[1][2:]]
    epoch_tensors_restart = [e["data"][0] for e in epoch_results_restart[1][2:]]

    for t, tr in zip(epoch_tensors, epoch_tensors_restart):
        assert torch.equal(t, tr)


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test takes around 45 sec and should be skipped in Azure CI")
def test_fast_forward_sampler_iterative_dataset():
    _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset(0, 1)


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test takes around 55 sec and should be skipped in Azure CI")
@RunIf(skip_windows=True)
def test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(
        _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset, args=(worldsize,), nprocs=worldsize
    )


def create_iterable_dataset(batch_size, num_workers, attr_name="iter_sampler"):
    dataset = RangeIterableDataset(range(50), num_workers=num_workers, batch_size=batch_size, attr_name=attr_name)
    return CaptureIterableDataset(dataset)


def create_dataloader():
    dataset = range(50)
    num_workers = 2
    batch_size = 8
    sampler = FastForwardSampler(SequentialSampler(dataset))
    sampler.setup(batch_size)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    dataloader.fast_forward_sampler = sampler

    loader_dict = {
        "a": [DataLoader(create_iterable_dataset(3, num_workers), num_workers=num_workers, batch_size=3), dataloader],
        "b": DataLoader(
            create_iterable_dataset(2, num_workers=1, attr_name="custom_sampler"), num_workers=0, batch_size=2
        ),
    }
    apply_to_collection(loader_dict, DataLoader, Trainer._add_sampler_metadata_collate)
    return CombinedLoader(loader_dict)


# Lightning will wrap the iterator within a prefect function as follow.
def prefetch_iterator(iterable: Iterable):
    it = iter(iterable)

    try:
        # the iterator may be empty from the beginning
        last = next(it)
    except StopIteration:
        return

    for val in it:
        # yield last and has next
        yield last, False, it
        last = val
    # yield last, no longer has next
    yield last, True, it


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test takes around 15 sec and should be skipped in Azure CI")
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
def test_combined_dataloader_state_dict_and_reload():
    """
    This test makes sure the CombinedLoader used in the condition of Lightning properly
    capture its children DataLoader states.
    """

    dataloader = create_dataloader()

    iter_dataloader = iter(prefetch_iterator(dataloader))
    num_batches_processed = 4
    for idx in range(1, num_batches_processed):
        _, _, prefetched_iterator = next(iter_dataloader)

        loader_iters = prefetched_iterator._loader_iters

        # when dealing with IterativeDataset,
        # the sampler state dict will be attached directly onto the iterator to simplify collection.

        if idx == 1:
            assert loader_iters["a"][0]._sampler_state_dict == [{"iter_sampler": {0: {"current_iteration": 3}}}]
            assert loader_iters["a"][1]._sampler_state_dict == []
            assert loader_iters["b"]._sampler_state_dict == [{"custom_sampler": {0: {"current_iteration": 2}}}]
        elif idx == 2:
            assert loader_iters["a"][0]._sampler_state_dict == [
                {"iter_sampler": {0: dict(current_iteration=3), 1: dict(current_iteration=3)}}
            ]
            assert loader_iters["a"][1]._sampler_state_dict == []
            assert loader_iters["b"]._sampler_state_dict == [{"custom_sampler": {0: {"current_iteration": 4}}}]
        else:
            assert loader_iters["a"][0]._sampler_state_dict == [
                {"iter_sampler": {0: dict(current_iteration=6), 1: dict(current_iteration=3)}}
            ]
            assert loader_iters["a"][1]._sampler_state_dict == []
            assert loader_iters["b"]._sampler_state_dict == [{"custom_sampler": {0: {"current_iteration": 6}}}]

    state_dict = dataloader.state_dict(num_batches_processed=3)

    expected = {
        "b": {"num_workers": 0, "previous_worker": None, "custom_sampler": {0: dict(current_iteration=6)}},
        "a": [
            {
                "num_workers": 2,
                "previous_worker": 1,
                "iter_sampler": {0: dict(current_iteration=6), 1: dict(current_iteration=3)},
            },
            {"num_workers": 0, "previous_worker": None, 0: dict(current_iteration=24)},
        ],
    }
    assert state_dict == expected

    dataloader = create_dataloader()
    apply_to_collection(dataloader, DataLoader, Trainer._add_sampler_metadata_collate)
    dataloader.load_state_dict(state_dict)

    iter_dataloader = iter(prefetch_iterator(dataloader))
    _, _, prefetched_iterator = next(iter_dataloader)

    loader_iters = prefetched_iterator._loader_iters

    assert loader_iters["a"][0]._sampler_state_dict == [
        {"num_workers": 2, "iter_sampler": {0: dict(current_iteration=6), 1: dict(current_iteration=6)}}
    ]
    assert loader_iters["a"][1]._sampler_state_dict == []
    assert loader_iters["b"]._sampler_state_dict == [
        {"num_workers": 0, "custom_sampler": {0: dict(current_iteration=8)}}
    ]

    state_dict = dataloader.state_dict(num_batches_processed=4)

    expected = {
        "a": [
            {
                "num_workers": 2,
                "previous_worker": 0,
                "iter_sampler": {0: dict(current_iteration=6), 1: dict(current_iteration=6)},
            },
            {"num_workers": 0, "previous_worker": None, 0: dict(current_iteration=32)},
        ],
        "b": {"num_workers": 0, "previous_worker": None, "custom_sampler": {0: dict(current_iteration=8)}},
    }

    assert state_dict == expected


def test_dataloader_to_state_dict_and_reload():
    """
    Note: Those utilities are used only with DataLoader wrapping a ``mapping`` based dataset.
    """

    def create_dataloader():
        dataset = range(50)
        batch_size = 8
        sampler = FastForwardSampler(SequentialSampler(dataset))
        sampler.setup(batch_size)

        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    dataloader = create_dataloader()
    iter_dataloader = iter(dataloader)
    _ = next(iter_dataloader)
    _ = next(iter_dataloader)

    state_dict = _dataloader_to_state_dict(dataloader, iter_dataloader)
    assert state_dict == {"num_workers": 0, "previous_worker": None, 0: {"current_iteration": 16}}

    dataloader = create_dataloader()
    dataloader = _dataloader_load_state_dict(dataloader, state_dict)
    iter_dataloader = iter(dataloader)
    _ = next(iter_dataloader)

    state_dict = _dataloader_to_state_dict(dataloader, iter_dataloader)
    assert state_dict == {"num_workers": 0, "previous_worker": None, 0: {"current_iteration": 24}}
