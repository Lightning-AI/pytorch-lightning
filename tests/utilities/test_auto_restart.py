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
import random as python_random
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from typing import List, Optional
from unittest import mock
from unittest.mock import ANY

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
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.trainer.progress import ReadyCompletedTracker
from pytorch_lightning.utilities.auto_restart import (
    _add_capture_metadata_collate,
    _dataloader_load_state_dict,
    _dataloader_to_state_dict,
    CaptureIterableDataset,
    CaptureMapDataset,
    FastForwardSampler,
    MergedIteratorState,
)
from pytorch_lightning.utilities.enums import AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import DataFetcher
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from tests.helpers.boring_model import BoringModel
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
    """This test ensures ``FastForwardSampler`` applied to ``BatchSampler`` correctly retrived the right next batch
    on restart."""
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
    """This test ensures ``FastForwardSampler`` applied to ``SequentialSampler`` correctly retrived the right next
    batch on restart."""
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
    """This test ensures ``FastForwardSampler`` applied to ``RandomSampler`` correctly retrived the right next
    batch on restart."""
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
def test_fast_forward_sampler_over_iterable_dataset(num_workers):
    """This test ensures ``FastForwardSampler`` and ``CaptureIterableDataset`` are properly being used to capture
    workers states."""
    batch_size = 3
    initial_seed = seed_everything(42)
    generator = torch.Generator()
    generator.manual_seed(initial_seed)
    dataset = RangeIterableDataset(range(20), num_workers, batch_size, True)
    dataset = CaptureIterableDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    _add_capture_metadata_collate(dataloader)

    iter_dataloader = iter(dataloader)
    batches = []
    for _ in range(5):
        batches.append(next(iter_dataloader))

    # restarting on batch_1 and getting 3 extra batches

    state_dict = {"iter_sampler": {}}
    for batch in batches[:2]:
        batch, _state_dict = batch["data"], batch[AutoRestartBatchKeys.PL_RESTART_META]
        for k, v in _state_dict.items():
            state_dict[k].update(v)

    assert len(state_dict["iter_sampler"]) == (num_workers if num_workers > 1 else 1)

    initial_seed = seed_everything(42)
    generator.manual_seed(initial_seed)
    dataset = RangeIterableDataset(range(20), num_workers, batch_size, state_dict=state_dict)
    dataset = CaptureIterableDataset(dataset)
    dataset.load_state_dict(state_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    _add_capture_metadata_collate(dataloader)

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
    """Make sure result logging works with DDP."""
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
        ):
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
    dataset = CaptureIterableDataset(dataset)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, generator=generator)
    _add_capture_metadata_collate(dataloader)

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
        assert 0 in epoch_results[0][2][AutoRestartBatchKeys.PL_RESTART_META]["iter_sampler"]  # worker id 0
        assert 1 in epoch_results[0][3][AutoRestartBatchKeys.PL_RESTART_META]["iter_sampler"]  # worker id 1
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
        batch, _state_dict = batch["data"], batch[AutoRestartBatchKeys.PL_RESTART_META]
        for k, v in _state_dict.items():
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

    dataset = CaptureIterableDataset(dataset)
    dataset.load_state_dict(state_dict)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, generator=generator)
    _add_capture_metadata_collate(dataloader)

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
    """Make sure result logging works with DDP."""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(
        _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset, args=(worldsize,), nprocs=worldsize
    )


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(max_torch="1.7")
def test_fault_tolerant_not_supported():
    assert not _fault_tolerant_training()


def create_iterable_dataset(batch_size, num_workers, attr_name="iter_sampler", wrap: bool = True):
    dataset = RangeIterableDataset(range(50), num_workers=num_workers, batch_size=batch_size, attr_name=attr_name)
    if wrap:
        dataset = CaptureIterableDataset(dataset)
    return dataset


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
    assert state_dict == {
        "num_workers": 0,
        "previous_worker": None,
        0: {"current_iteration": 16},
    }

    dataloader = create_dataloader()
    dataloader = _dataloader_load_state_dict(dataloader, state_dict)
    iter_dataloader = iter(dataloader)
    _ = next(iter_dataloader)

    state_dict = _dataloader_to_state_dict(dataloader, iter_dataloader)
    assert state_dict == {
        "num_workers": 0,
        "previous_worker": None,
        0: {"current_iteration": 24},
    }


@RunIf(min_torch="1.7.0")
@pytest.mark.parametrize("use_fault_tolerant", ["0", "1"])
def test_data_loading_wraps_dataset_and_samplers(use_fault_tolerant, tmpdir):
    """This test ensures the dataset and sampler are properly wrapped when fault tolerant is enabled."""

    class CustomBatchSampler(BatchSampler):
        pass

    dataset = range(50)

    class TestModel(BoringModel):
        def train_dataloader(self):
            return {
                "a": [
                    DataLoader(create_iterable_dataset(3, 1, wrap=False), num_workers=0, batch_size=3),
                    DataLoader(dataset, batch_size=8),
                    DataLoader(
                        dataset,
                        batch_sampler=CustomBatchSampler(SequentialSampler(dataset), batch_size=8, drop_last=False),
                    ),
                ],
                "b": DataLoader(
                    create_iterable_dataset(2, num_workers=1, attr_name="custom_sampler", wrap=False),
                    num_workers=0,
                    batch_size=2,
                ),
            }

        def training_step(self, batch, batch_idx):
            assert batch == {
                "a": [ANY, ANY, ANY],
                "b": ANY,
            }

        def validation_step(self, batch, batch_idx):
            assert isinstance(batch, torch.Tensor)

        validation_epoch_end = None

    class Check(Callback):
        def on_train_batch_start(self, trainer, *_) -> None:
            loaders = trainer.train_dataloader.loaders
            if use_fault_tolerant == "1":
                assert isinstance(loaders["a"][0].loader.dataset, CaptureIterableDataset)
                assert isinstance(loaders["a"][1].loader.sampler, FastForwardSampler)
                assert isinstance(loaders["a"][1].loader.dataset, CaptureMapDataset)
                assert isinstance(loaders["a"][2].loader.batch_sampler, FastForwardSampler)
                assert isinstance(loaders["a"][2].loader.dataset, CaptureMapDataset)
                assert isinstance(loaders["b"].loader.dataset, CaptureIterableDataset)
            else:
                assert isinstance(loaders["a"][0].loader.dataset, RangeIterableDataset)
                assert isinstance(loaders["a"][1].loader.sampler, SequentialSampler)
                assert not isinstance(loaders["a"][1].loader.dataset, CaptureMapDataset)
                assert isinstance(loaders["a"][2].loader.batch_sampler, CustomBatchSampler)
                assert not isinstance(loaders["a"][2].loader.dataset, CaptureMapDataset)
                assert isinstance(loaders["b"].loader.dataset, RangeIterableDataset)

    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": use_fault_tolerant}):
        model = TestModel()
        model.training_epoch_end = None
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=1, callbacks=Check())
        trainer.fit(model)


class SequentialGetItemDataset(Dataset):
    def __init__(self, length, *_):
        self.len = length

    def __getitem__(self, index):
        return torch.tensor([index]).float()

    def __len__(self):
        return self.len


class RandomGetItemDataset(Dataset):
    """A dataset with random elements generated using global rng from torch, numpy and python."""

    def __init__(self, length, size):
        self.size = size
        self.len = length

    def __getitem__(self, index):
        t = torch.rand(self.size)
        n = torch.from_numpy(np.random.rand(self.size))
        p = torch.tensor([python_random.random() for _ in range(self.size)])
        sample = (index + (t + n + p) / 10).float()
        return sample

    def __len__(self):
        return self.len


# TODO: test with `RandomGeneratorGetItemDataset`
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
@pytest.mark.parametrize(
    "dataset_class",
    [
        SequentialGetItemDataset,
        RandomGetItemDataset,
        # RandomGeneratorGetItemDataset,
    ],
)
@pytest.mark.parametrize("num_workers", [0])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_dataset_rng_states_restart(dataset_class, num_workers, batch_size):
    """Test that the sequence of batches coming from a random number generator continues with the correct sequence
    after reloading the state."""

    def create_dataset_sampler():
        dset = CaptureMapDataset(dataset_class(16, 8))
        random_sampler = RandomSampler(dset, generator=torch.Generator())
        return dset, random_sampler

    def create_dataloader_sampler(dset, sampler):
        sampler = FastForwardSampler(sampler)
        sampler.setup(batch_size)
        dl = DataLoader(dset, num_workers=num_workers, sampler=sampler, batch_size=batch_size)
        _add_capture_metadata_collate(dl)
        return dl, sampler

    def fetch(fetcher, prefetch_iter, num_batches_fetched):
        batch, _ = next(prefetch_iter)

        state: List[MergedIteratorState] = fetcher.state
        assert len(state) == 1
        assert isinstance(state[0], MergedIteratorState)

        assert len(fetcher.dataloader_iter.cache_states) == 1
        if num_workers == 0:
            assert state[0].state[0].num_batches_fetched == num_batches_fetched
        return state

    dataset, random_sampler = create_dataset_sampler()
    dataloader, ff_sampler = create_dataloader_sampler(dataset, random_sampler)

    fetcher = DataFetcher()
    fetcher.setup(dataloader)
    prefetch_iter = iter(fetcher)

    # fetch 4 batches
    fetch(fetcher, prefetch_iter, 1)
    fetch(fetcher, prefetch_iter, 2)
    fetch(fetcher, prefetch_iter, 3)

    # (A) capture the state after fetching 4 batches
    state = fetch(fetcher, prefetch_iter, 4)
    state = deepcopy(state[0])

    # (B) simulate 2 additional batches
    batch05, _ = next(prefetch_iter)
    batch06, _ = next(prefetch_iter)

    # start reloading
    dataset, random_sampler = create_dataset_sampler()
    dataloader, ff_sampler = create_dataloader_sampler(dataset, random_sampler)

    # load the state dict saved at (A)
    ff_sampler.load_state_dict(state.sampler_states)
    dataset.load_state_dict(state.dataset_states, latest_worker_id=state.latest_worker_id, num_workers=num_workers)

    prefetcher = DataFetcher()
    prefetcher.setup(dataloader)
    prefetch_iter = iter(prefetcher)

    # fetch 2 random batches, these should match exactly the batches seen at (B)
    batch05_restart, _ = next(prefetch_iter)
    batch06_restart, _ = next(prefetch_iter)

    assert torch.equal(batch05, batch05_restart)
    assert torch.equal(batch06, batch06_restart)


class CustomException(Exception):
    pass


class SequentialIterableDataset(IterableDataset):
    def __init__(self, length, *_):
        self.len = length
        self.sampler = SequentialSampler(range(self.len))

    def __iter__(self):
        self.sampler_iter = iter(self.sampler)
        return self

    def __next__(self):
        indices = next(self.sampler_iter)
        return torch.tensor([indices]).float()


class SequentialDictIterableDataset(SequentialIterableDataset):
    def __next__(self):
        indices = next(self.sampler_iter)
        return {"data": torch.tensor([indices]).float()}


class TestModel(LightningModule):
    def __init__(self, fail_on_step: int = -1):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)
        self.seen_batches = []
        self.fail_on_step = fail_on_step

    def training_step(self, batch, batch_idx):
        if self.global_step == self.fail_on_step:
            raise CustomException()
        batch = batch["data"] if isinstance(batch, dict) else batch
        self.seen_batches.append(torch.stack(batch) if isinstance(batch, list) else batch)
        loss = sum(self.layer(b).sum() for b in batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def _run_training(trainer_kwargs, dataset_classes, fail_on_step: int = -1):
    seed_everything(1)
    train_dataloader = [
        DataLoader(dataset_class(3, 1), batch_size=1, num_workers=0) for dataset_class in dataset_classes
    ]
    train_dataloader = train_dataloader[0] if len(train_dataloader) == 1 else train_dataloader
    model = TestModel(fail_on_step=fail_on_step)
    trainer = Trainer(**trainer_kwargs)
    with suppress(CustomException):
        trainer.fit(model, train_dataloader=train_dataloader)
    return model.seen_batches, model.parameters()


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
@pytest.mark.parametrize("use_faulty_optimizer", [False, True])
def test_fault_tolerant_supported(use_faulty_optimizer, tmpdir):

    """This test asserts a fault tolerant checkpoint is generated during failure on training step, but not during
    optimizer.step execution."""

    class FaultyOptimizer(torch.optim.SGD):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.counter = 0

        def _closure(self, loss):
            def fn():
                return loss

            return fn

        def step(self, closure) -> Optional[float]:
            loss = closure()
            if self.counter == 2:
                raise CustomException
            self.counter += 1
            return super().step(closure=self._closure(loss))

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            if not use_faulty_optimizer and batch_idx == 2:
                raise CustomException
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            if not use_faulty_optimizer:
                return super().configure_optimizers()
            else:
                return FaultyOptimizer(self.parameters(), lr=0.001)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=3, limit_val_batches=3)
    with suppress(CustomException):
        trainer.fit(TestModel())

    checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path) == (not use_faulty_optimizer)


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
@pytest.mark.parametrize(
    "dataset_classes",
    [
        # single training dataset
        [RandomGetItemDataset],
        [SequentialIterableDataset],
        [SequentialDictIterableDataset],
        # multiple training datasets (combinded dataloader)
        [SequentialGetItemDataset, SequentialIterableDataset],
        [SequentialIterableDataset, SequentialIterableDataset],
        # [RandomGetItemDataset, RandomGetItemDataset],  # TODO: support in the future
    ],
)
@pytest.mark.parametrize("multiple_trainloader_mode", ["min_size", "max_size_cycle"])
def test_dataset_rng_states_restart_with_lightning(tmpdir, dataset_classes, multiple_trainloader_mode):
    """Test that the Trainer can resume from a failed run in the case of several types of datasets."""
    trainer_kwargs = dict(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        progress_bar_refresh_rate=0,
        multiple_trainloader_mode=multiple_trainloader_mode,
    )

    all_batches, weights0 = _run_training(trainer_kwargs, dataset_classes)
    all_batches = torch.stack(all_batches)
    assert len(all_batches) == 9

    # Simulate 1st failure
    complete_batches, _ = _run_training(trainer_kwargs, dataset_classes, fail_on_step=4)
    assert len(complete_batches) == 4

    checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path)

    # Resume after failure
    trainer_kwargs.update(resume_from_checkpoint=checkpoint_path)
    resumed_batches, weights1 = _run_training(trainer_kwargs, dataset_classes, fail_on_step=-1)
    assert len(resumed_batches) == 5

    # the resumed batches should match the batches of the successful training
    all_batches_resumed = torch.stack(complete_batches + resumed_batches)
    assert len(all_batches_resumed) == 9
    assert torch.equal(all_batches, all_batches_resumed)

    # the final weights of a resumed training should equal the weights of an uninterrupted training
    for w0, w1 in zip(weights0, weights1):
        assert w0 is not w1
        assert torch.allclose(w0, w1)


class ValidationLoopTestModel(LightningModule):
    def __init__(self, fail_on_dataloader: int = -1, val_check_interval: float = 0.5):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)
        self.training_seen_batches = []
        self.validation_seen_batches = defaultdict(list)
        self.fail_on_dataloader = fail_on_dataloader
        self.failing_batch_idx: Optional[int] = None
        self.failing_dataloader_int: Optional[int] = None
        self.val_check_interval = val_check_interval

    def training_step(self, batch, batch_idx):
        print("training_step")
        batch = batch["data"] if isinstance(batch, dict) else batch
        self.training_seen_batches.append(torch.stack(batch) if isinstance(batch, list) else batch)
        loss = sum(self.layer(b).sum() for b in batch)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_int: int = 0):
        print(
            self.trainer.sanity_checking,
            dataloader_int,
            batch_idx,
            self.trainer.current_epoch,
            self.trainer.global_step,
        )

        loss = sum(self.layer(b).sum() for b in batch)

        if self.trainer.sanity_checking:
            return loss

        # failure on first batch of dataloader_idx once global_step is 1
        if (
            self.fail_on_dataloader >= 0
            and self.fail_on_dataloader == dataloader_int
            and batch_idx == 1
            and (self.trainer.global_step == 3 if self.val_check_interval == 1.0 else self.trainer.global_step == 1)
        ):
            self.failing_batch_idx = batch_idx
            self.failing_dataloader_int = dataloader_int
            raise CustomException

        self.validation_seen_batches[dataloader_int].append(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
@pytest.mark.parametrize(
    "dataset_classes",
    [
        # single training dataset
        # [[RandomGetItemDataset], [RandomGetItemDataset]],
        [[RandomGetItemDataset], [RandomGetItemDataset]],
        # [[RandomGetItemDataset], [RandomGetItemDataset, RandomGetItemDataset]],
        # [SequentialIterableDataset],
        # [SequentialDictIterableDataset],
        # [SequentialGetItemDataset, SequentialIterableDataset],
        # [SequentialIterableDataset, SequentialIterableDataset],
    ],
)
@pytest.mark.parametrize("val_check_interval", [0.5, 1.0])
def test_auto_restart_within_validation_loop(dataset_classes, val_check_interval, tmpdir):

    seed_everything(42)
    num_samples = 4
    train_dataset_classes, validation_dataset_classes = dataset_classes
    train_dataloader = [
        DataLoader(dataset_class(num_samples, 1), batch_size=1, num_workers=0)
        for dataset_class in train_dataset_classes
    ]
    val_dataloaders = [
        DataLoader(dataset_class(num_samples, 1), batch_size=1, num_workers=0)
        for dataset_class in validation_dataset_classes
    ]

    # enable `num_sanity_val_steps=2`
    trainer_kwargs = dict(
        default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval, num_sanity_val_steps=0
    )

    model = ValidationLoopTestModel()
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)

    verif_train_batches = model.training_seen_batches
    verif_valid_batches = model.validation_seen_batches

    num_validation_loaders = len(validation_dataset_classes)

    assert len(verif_train_batches) == num_samples
    assert len(verif_valid_batches) == num_validation_loaders
    for batch in verif_valid_batches.values():
        assert len(batch) == (1 / val_check_interval) * num_samples

    seed_everything(42)
    fail_on_dataloader = num_validation_loaders - 1

    model = ValidationLoopTestModel(fail_on_dataloader=fail_on_dataloader, val_check_interval=val_check_interval)
    trainer = Trainer(**trainer_kwargs)
    with suppress(CustomException):
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)

    pre_fail_train_batches = model.training_seen_batches
    pre_fail_valid_batches = model.validation_seen_batches

    assert len(pre_fail_train_batches) == 4 if val_check_interval == 1.0 else 2
    # assert verif_train_batches[:2] == pre_fail_train_batches

    if num_validation_loaders == 2:
        assert len(pre_fail_valid_batches[0]) == 4
        assert len(pre_fail_valid_batches[1]) == 1
    else:
        assert len(pre_fail_valid_batches[0]) == 1

    assert model.failing_batch_idx == 1
    assert model.failing_dataloader_int == fail_on_dataloader

    trainer_kwargs["resume_from_checkpoint"] = checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(checkpoint_path)["loops"]["fit_loop"]

    shift = 2 if val_check_interval == 1.0 else 0
    assert checkpoint["epoch_loop.batch_progress"]["total"] == {
        "ready": 2 + shift,
        "completed": 2 + shift,
        "started": 2 + shift,
        "processed": 2 + shift,
    }
    assert checkpoint["epoch_loop.batch_progress"]["current"] == {
        "ready": 2 + shift,
        "completed": 2 + shift,
        "started": 2 + shift,
        "processed": 2 + shift,
    }

    shift = 1 if num_validation_loaders == 2 else 0
    assert checkpoint["epoch_loop.val_loop.dataloader_progress"]["total"] == {
        "ready": 1 + shift,
        "completed": shift,
    }

    assert checkpoint["epoch_loop.val_loop.dataloader_progress"]["current"] == {
        "ready": 1 + shift,
        "completed": 0 + shift,
    }

    trainer = Trainer(**trainer_kwargs)
    model = ValidationLoopTestModel()
    assert model.training_seen_batches == []
    assert len(model.validation_seen_batches) == 0
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)

    dataloader_progress = trainer.fit_loop.epoch_loop.val_loop.dataloader_progress

    dataloader_progress.total = ReadyCompletedTracker(
        ready=num_validation_loaders + 1, completed=num_validation_loaders
    )
    dataloader_progress.current = ReadyCompletedTracker(ready=num_validation_loaders, completed=num_validation_loaders)

    post_fail_train_batches = model.training_seen_batches
    post_fail_valid_batches = model.validation_seen_batches

    assert len(verif_train_batches) == len(pre_fail_train_batches) + len(post_fail_train_batches)
    assert len(verif_valid_batches[0]) == len(pre_fail_valid_batches[0]) + len(post_fail_valid_batches[0])
    if num_validation_loaders == 2:
        assert len(verif_valid_batches[1]) == len(pre_fail_valid_batches[1]) + len(post_fail_valid_batches[1])
