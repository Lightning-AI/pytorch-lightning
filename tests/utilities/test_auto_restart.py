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
import inspect
import math
import os
import random
import random as python_random
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict
from typing import Iterator, List, Optional
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
from torch.utils.data.sampler import Sampler

import tests.helpers.utils as tutils
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.trainer.states import RunningStage, TrainerState
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.auto_restart import (
    _add_capture_metadata_collate,
    _collect_states_on_rank_zero_over_collection,
    _MultiProcessingDataLoaderIterStateful,
    _patch_dataloader_get_iterators,
    _reload_dataloader_state_dict,
    _rotate_worker_indices,
    _SingleProcessDataLoaderIterStateful,
    _teardown_dataloader_get_iterators,
    _validate_fault_tolerant_automatic,
    CaptureIterableDataset,
    CaptureMapDataset,
    FastForwardSampler,
    MergedIteratorState,
)
from pytorch_lightning.utilities.enums import _FaultTolerantMode, AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import DataFetcher
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from tests.helpers.boring_model import BoringModel, RandomDataset
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
    """This test ensures ``FastForwardSampler`` applied to ``BatchSampler`` correctly retrieved the right next
    batch on restart."""
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
    """This test ensures ``FastForwardSampler`` applied to ``SequentialSampler`` correctly retrieved the right next
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


def test_fast_forward_on_random_sampler():
    """This test ensures ``FastForwardSampler`` applied to ``RandomSampler`` correctly retrieved the right next
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


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize(
    "num_workers", [0, pytest.param(1, marks=RunIf(slow=True)), pytest.param(2, marks=RunIf(slow=True))]
)
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


@RunIf(skip_windows=True, slow=True)
def test_fast_forward_sampler_with_distributed_sampler():
    """Make sure result logging works with DDP."""
    tutils.set_random_main_port()
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


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(slow=True)
def test_fast_forward_sampler_iterative_dataset():
    _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset(0, 1)


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(skip_windows=True, slow=True)
def test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset():
    """Make sure result logging works with DDP."""
    tutils.set_random_main_port()
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


@mock.patch("pytorch_lightning.trainer.connectors.data_connector._validate_fault_tolerant_automatic")
@pytest.mark.parametrize("use_fault_tolerant", ["0", "1"])
def test_data_loading_wraps_dataset_and_samplers(_, tmpdir, use_fault_tolerant):
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
@pytest.mark.parametrize(
    "dataset_class",
    [
        SequentialGetItemDataset,
        RandomGetItemDataset,
        # RandomGeneratorGetItemDataset,
    ],
)
@pytest.mark.parametrize("num_workers", [0, pytest.param(2, marks=RunIf(slow=True))])
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
        _ = next(prefetch_iter)

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
    batch05 = next(prefetch_iter)
    batch06 = next(prefetch_iter)

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
    batch05_restart = next(prefetch_iter)
    batch06_restart = next(prefetch_iter)

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


def _run_training(trainer_kwargs, dataset_classes, fail_on_step: int = -1, ckpt_path=None):
    seed_everything(1)
    train_dataloader = [
        DataLoader(dataset_class(3, 1), batch_size=1, num_workers=0) for dataset_class in dataset_classes
    ]
    train_dataloader = train_dataloader[0] if len(train_dataloader) == 1 else train_dataloader
    model = TestModel(fail_on_step=fail_on_step)
    trainer = Trainer(**trainer_kwargs)
    with suppress(CustomException):
        trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=ckpt_path)
    return model.seen_batches, model.parameters()


@mock.patch("pytorch_lightning.trainer.connectors.data_connector._validate_fault_tolerant_automatic")
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize(
    "dataset_classes",
    [
        # single training dataset
        [RandomGetItemDataset],
        [SequentialIterableDataset],
        [SequentialDictIterableDataset],
        # multiple training datasets (combined dataloader)
        [SequentialGetItemDataset, SequentialIterableDataset],
        [SequentialIterableDataset, SequentialIterableDataset],
        # [RandomGetItemDataset, RandomGetItemDataset],  # TODO: support in the future
    ],
)
@pytest.mark.parametrize("multiple_trainloader_mode", ["min_size", "max_size_cycle"])
def test_dataset_rng_states_restart_with_lightning(_, tmpdir, dataset_classes, multiple_trainloader_mode):
    """Test that the Trainer can resume from a failed run in the case of several types of datasets."""
    trainer_kwargs = dict(
        default_root_dir=tmpdir,
        max_epochs=3,
        enable_progress_bar=False,
        enable_model_summary=False,
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
    resumed_batches, weights1 = _run_training(
        trainer_kwargs, dataset_classes, fail_on_step=-1, ckpt_path=checkpoint_path
    )
    assert len(resumed_batches) == 5

    # the resumed batches should match the batches of the successful training
    all_batches_resumed = torch.stack(complete_batches + resumed_batches)
    assert len(all_batches_resumed) == 9
    assert torch.equal(all_batches, all_batches_resumed)

    # the final weights of a resumed training should equal the weights of an uninterrupted training
    for w0, w1 in zip(weights0, weights1):
        assert w0 is not w1
        assert torch.allclose(w0, w1)


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize(
    ["train_datasets", "val_datasets"],
    [
        ([RandomGetItemDataset], [RandomGetItemDataset]),
        ([RandomGetItemDataset], [RandomGetItemDataset, RandomGetItemDataset]),
    ],
)
@pytest.mark.parametrize(
    "val_check_interval",
    [
        pytest.param(
            0.5,
            marks=pytest.mark.xfail(
                reason=(
                    "TODO: the `train_dataloader` random state overrides the validation state when restarting training"
                )
            ),
        ),
        1.0,
    ],
)
def test_auto_restart_within_validation_loop(train_datasets, val_datasets, val_check_interval, tmpdir):
    n_val_dataloaders = len(val_datasets)
    stop_dataloader = n_val_dataloaders - 1
    stop_batch = 1

    class ValidationLoopTestModel(LightningModule):
        def __init__(self, should_fail):
            super().__init__()
            self.layer = torch.nn.Linear(1, 2)
            self.should_fail = should_fail
            self.training_batches = []
            self.validation_batches = defaultdict(list)

        def step(self, batch):
            return sum(self.layer(b).sum() for b in batch)

        def training_step(self, batch, batch_idx):
            self.training_batches.append(batch)
            return self.step(batch)

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            if self.should_fail and stop_dataloader == dataloader_idx and batch_idx == stop_batch:
                raise CustomException
            self.validation_batches[dataloader_idx].append(batch)
            return self.step(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

        def train_dataloader(self):
            return [DataLoader(cls(4, 1)) for cls in train_datasets]

        def val_dataloader(self):
            return [DataLoader(cls(4, 1)) for cls in val_datasets]

    def run(should_fail, resume):
        if not resume:
            seed_everything(42)

        model = ValidationLoopTestModel(should_fail)

        ckpt_path = str(tmpdir / ".pl_auto_save.ckpt") if resume else None
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            val_check_interval=val_check_interval,
            num_sanity_val_steps=0,
        )
        if should_fail:
            with pytest.raises(CustomException):
                trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        return model.training_batches, model.validation_batches

    total_train_batches, total_val_batches = run(should_fail=False, resume=False)
    pre_fail_train_batches, pre_fail_val_batches = run(should_fail=True, resume=False)
    post_fail_train_batches, post_fail_val_batches = run(should_fail=False, resume=True)

    torch.testing.assert_allclose(total_train_batches, pre_fail_train_batches + post_fail_train_batches)
    for k in total_val_batches:
        torch.testing.assert_allclose(total_val_batches[k], pre_fail_val_batches[k] + post_fail_val_batches[k])


class TestAutoRestartModelUnderSignal(BoringModel):
    def __init__(self, should_signal: bool, failure_on_step: bool, failure_on_training: bool, on_last_batch: bool):
        super().__init__()
        self.should_signal = should_signal
        self.failure_on_step = failure_on_step
        self.failure_on_training = failure_on_training
        self.on_last_batch = on_last_batch
        self.seen_train_batches = []

    def _signal(self):
        if self.should_signal:
            # simulate `os.kill(os.getpid(), signal.SIGTERM)`
            self.trainer._terminate_gracefully = True

    def training_step(self, batch, batch_idx):
        self.seen_train_batches.append(batch)
        should_signal = self.trainer.fit_loop.epoch_loop._is_training_done if self.on_last_batch else batch_idx == 2
        if self.failure_on_step and self.failure_on_training and should_signal:
            self._signal()
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        should_signal = (
            self.trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.is_last_batch
            if self.on_last_batch
            else batch_idx == 2
        )
        if self.failure_on_step and not self.failure_on_training and should_signal:
            self._signal()
        return super().validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs) -> None:
        if not self.failure_on_step and self.failure_on_training:
            self._signal()

    def validation_epoch_end(self, outputs) -> None:
        if not self.failure_on_step and not self.failure_on_training:
            self._signal()

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 4))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 4))


def _fit_model(
    tmpdir, should_signal, val_check_interval, failure_on_step, failure_on_training, on_last_batch, status=None
):
    seed_everything(42)
    model = TestAutoRestartModelUnderSignal(should_signal, failure_on_step, failure_on_training, on_last_batch)

    trainer_kwargs = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
    )

    class ExitGracefullyException(Exception):
        pass

    class TestTrainer(Trainer):
        def _exit_gracefully_on_signal(self) -> None:
            if not _fault_tolerant_training() or not self._should_terminate_gracefully():
                return
            caller = inspect.stack()[1]
            class_name = caller[0].f_locals["self"].__class__.__name__
            raise ExitGracefullyException(f"Exiting gracefully on {class_name}:{caller.function}")

    trainer = TestTrainer(**trainer_kwargs)
    if should_signal:
        with pytest.raises(ExitGracefullyException, match=status):
            trainer.fit(model)
    else:
        trainer.fit(model)
    assert trainer._terminate_gracefully == should_signal

    return model


@pytest.mark.parametrize("on_last_batch", [False, True])
@pytest.mark.parametrize("val_check_interval", [0.5, 1.0])
@pytest.mark.parametrize("failure_on_training", [False, True])
@pytest.mark.parametrize("failure_on_step", [False, True])
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(skip_windows=True)
def test_auto_restart_under_signal(on_last_batch, val_check_interval, failure_on_training, failure_on_step, tmpdir):
    """This test asserts that if a signal is being sent during the training / validation phase, the model should
    restart in a reproducible way."""

    model_total = _fit_model(tmpdir, False, val_check_interval, failure_on_step, failure_on_training, on_last_batch)

    if failure_on_step:
        if on_last_batch:
            if failure_on_training:
                # Breaking on first validation batch.
                # This is done to capture the random state of the validation dataloader.
                status = "EvaluationEpochLoop:advance"
            else:
                # when breaking on last batch of validation, we should exist on `run_end` val_check_interval == 1.0
                status = "FitLoop:on_advance_end" if val_check_interval == 1.0 else "TrainingEpochLoop:on_advance_end"
        else:
            status = "TrainingEpochLoop:on_advance_end" if failure_on_training else "EvaluationEpochLoop:advance"
    else:
        if val_check_interval == 1.0:
            status = "FitLoop:on_advance_end"
        else:
            # `training_epoch_end` happens after `validation_epoch_end` since Lightning v1.4
            status = "FitLoop:on_advance_end" if failure_on_training else "TrainingEpochLoop:on_advance_end"

    model_signaled = _fit_model(
        tmpdir, True, val_check_interval, failure_on_step, failure_on_training, on_last_batch, status=status
    )
    # we saved a ft-checkpoint
    signaled_ckpt_path = str(tmpdir / ".pl_auto_save.ckpt")
    assert os.path.exists(signaled_ckpt_path)
    # load for later as the next fit call will delete it
    checkpoint = torch.load(signaled_ckpt_path)["loops"]["fit_loop"]

    model_restarted = _fit_model(tmpdir, False, val_check_interval, failure_on_step, failure_on_training, on_last_batch)

    # check the batches
    actual = torch.cat(model_signaled.seen_train_batches + model_restarted.seen_train_batches)
    expected = torch.cat(model_total.seen_train_batches)
    assert torch.equal(actual, expected)

    # FIXME: why `on_last_batch` doesn't work ?
    if failure_on_step and failure_on_training and not on_last_batch:
        assert not torch.equal(model_total.layer.weight, model_signaled.layer.weight)
    assert torch.equal(model_restarted.layer.weight, model_total.layer.weight)

    p = checkpoint["epoch_loop.batch_progress"]
    if p["is_last_batch"] and p["current"]["completed"] == 4:
        assert "dataloader_state_dict" not in checkpoint["epoch_loop.state_dict"]
    else:
        assert "dataloader_state_dict" in checkpoint["epoch_loop.state_dict"]

    state_dict = checkpoint["epoch_loop.val_loop.epoch_loop.state_dict"]
    p = checkpoint["epoch_loop.val_loop.epoch_loop.batch_progress"]
    if (p["is_last_batch"] and p["current"]["completed"] == 4) or p["current"]["ready"] == 0:
        assert "dataloader_state_dict" not in state_dict
    else:
        assert "dataloader_state_dict" in state_dict


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_validate_fault_tolerant(tmpdir):
    def data():
        return range(10)

    def dataloader():
        return DataLoader(data())

    _validate_fault_tolerant_automatic(dataloader(), RunningStage.TRAINING)

    dataloaders = CombinedLoader([dataloader(), dataloader()])
    with pytest.raises(ValueError, match="Fault-tolerance supports only a single dataloader."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    dataloaders = CombinedLoader([dataloader(), dataloader()], mode="max_size_cycle")
    with pytest.raises(ValueError, match="Fault-tolerance supports only a single dataloader."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    dataloaders = [dataloader(), dataloader()]
    with pytest.raises(ValueError, match="Fault-tolerance supports only a single dataloader."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    _validate_fault_tolerant_automatic(dataloaders, RunningStage.VALIDATING)

    dataloaders = [DataLoader(data(), sampler=DistributedSampler(data(), num_replicas=2, rank=0, shuffle=True))]
    with pytest.raises(TypeError, match="A `DistributedSampler` sampler shuffle attribute is set to True."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    dataloaders = [DataLoader(data(), sampler=DistributedSampler(data(), num_replicas=2, rank=0, shuffle=False))]
    _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    dataset = SequentialGetItemDataset(2)
    dataloaders = [
        DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)),
        DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)),
    ]
    with pytest.raises(ValueError, match="Fault-tolerance supports only a single dataloader."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    dataloaders = [
        DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)),
        DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)),
    ]
    with pytest.raises(ValueError, match="Fault-tolerance supports only a single."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.TRAINING)

    dataloaders = [
        DataLoader(dataset, sampler=RandomSampler(dataset)),
        DataLoader(dataset, sampler=SequentialSampler(dataset)),
    ]

    with pytest.raises(TypeError, match="Only `SequentialSampler` is supported."):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.VALIDATING)

    class CustomRandomSampler(RandomSampler):
        pass

    dl = DataLoader(data(), sampler=CustomRandomSampler(data()))
    with pytest.raises(TypeError, match="RandomSampler"):
        _validate_fault_tolerant_automatic(dl, RunningStage.TRAINING)

    class CustomBatchSampler(BatchSampler):
        pass

    sampler = Sampler(data())
    batch_sampler = CustomBatchSampler(sampler, 2, False)
    dl = DataLoader(data(), batch_sampler=batch_sampler)
    with pytest.raises(TypeError, match="BatchSampler"):
        _validate_fault_tolerant_automatic(dl, RunningStage.TRAINING)

    class CustomIterable(IterableDataset):
        pass

    iterable_dataloader = DataLoader(CustomIterable())
    with pytest.raises(AttributeError, match="without `__next__` method"):
        _validate_fault_tolerant_automatic(iterable_dataloader, RunningStage.TRAINING)

    class CustomIterable(IterableDataset):
        def __next__(self):
            return torch.tensor(0)

    iterable_dataloader = DataLoader(CustomIterable())
    with pytest.raises(TypeError, match="IterableDataset without a sampler as attribute"):
        _validate_fault_tolerant_automatic(iterable_dataloader, RunningStage.TRAINING)

    class CustomIterable(IterableDataset):
        def __init__(self):
            super().__init__()
            self.sampler = CustomRandomSampler(data())

        def __next__(self):
            return torch.tensor(0)

    iterable_dataloader = DataLoader(CustomIterable())
    with pytest.raises(TypeError, match="RandomSampler"):
        _validate_fault_tolerant_automatic(iterable_dataloader, RunningStage.TRAINING)

    dataloaders = [iterable_dataloader, DataLoader(CustomIterable())]
    with pytest.raises(TypeError, match="RandomSampler"):
        _validate_fault_tolerant_automatic(dataloaders, RunningStage.VALIDATING)


def test_rotate_worker_indices():
    """This test ensures `worker_id` are rotated properly depending on which one was the latest."""
    state_dict = {0: 0, 1: 1}
    assert _rotate_worker_indices(state_dict, 0, 2) == {0: 1, 1: 0}
    assert _rotate_worker_indices(state_dict, 1, 2) == {0: 0, 1: 1}

    with pytest.raises(MisconfigurationException, match="The `latest_worker_id` should be within"):
        _rotate_worker_indices(state_dict, 2, 2)

    with pytest.raises(MisconfigurationException, match="The `state` should contain"):
        _rotate_worker_indices(state_dict, 2, 3)


def test_fault_tolerant_mode_enum():
    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "0"}):
        assert _FaultTolerantMode.DISABLED == _FaultTolerantMode.detect_current_mode()
        assert not TrainerState()._fault_tolerant_mode.is_enabled

    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"}):
        assert _FaultTolerantMode.AUTOMATIC == _FaultTolerantMode.detect_current_mode()
        assert TrainerState()._fault_tolerant_mode.is_automatic

    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "MANUAL"}):
        assert _FaultTolerantMode.MANUAL == _FaultTolerantMode.detect_current_mode()
        assert TrainerState()._fault_tolerant_mode.is_manual

    with pytest.raises(
        MisconfigurationException, match="The environment flag `PL_FAULT_TOLERANT_TRAINING` should be either"
    ):
        with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "3"}):
            _FaultTolerantMode.detect_current_mode()


class StatefulRandomSampler(RandomSampler):

    counter = 0

    def state_dict(self):
        self.counter += 1
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]


class StatefulRandomDataset(RandomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def __getitem__(self, index):
        self.counter += 1
        return super().__getitem__(index)

    def state_dict(self):
        info = get_worker_info()
        if info:
            return {info.id: {"counter": self.counter}}
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict[0]["counter"]


def test_collect_states_with_collection():
    state = {"state": 0}
    collection = [{"a": state, "b": [{"a": state}]}]
    generated = _collect_states_on_rank_zero_over_collection(collection)
    assert generated == [{"a": {0: state}, "b": [{"a": {0: state}}]}]


# FIXME(@tchaton): >0 num_workers failing
@pytest.mark.parametrize("num_workers", [0, pytest.param(2, marks=[RunIf(slow=True), pytest.mark.xfail()])])
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "2"})
def test_stateful_workers(num_workers):
    seed_everything(42)

    _get_iterator_fn = DataLoader._get_iterator
    _patch_dataloader_get_iterators()
    assert DataLoader._ori_get_iterator is not None

    data_fetcher = DataFetcher()
    dataset = StatefulRandomDataset(1, 64)
    dataloader = DataLoader(dataset, sampler=StatefulRandomSampler(dataset), num_workers=num_workers)

    with pytest.raises(MisconfigurationException, match="A stateful iterator should be used"):
        iter(dataloader)

    # This would attach the `data_fetcher` to the DataLoader.
    data_fetcher.setup(dataloader)

    data_fetcher_iter = iter(data_fetcher)

    dataloader_iter = data_fetcher.dataloader_iter
    worker_type = _SingleProcessDataLoaderIterStateful if num_workers == 0 else _MultiProcessingDataLoaderIterStateful
    assert isinstance(dataloader_iter, worker_type)

    next(data_fetcher_iter)

    reloaded_state = deepcopy(data_fetcher.dataloader_iter.state)
    state = reloaded_state.state
    assert state[0].dataset_state == {0: {"counter": 1}}
    assert state[0].sampler_state["sampler"] == {"counter": 1}

    next(data_fetcher_iter)
    previous_state = data_fetcher.dataloader_iter.previous_state.state
    state = data_fetcher.dataloader_iter.state.state
    assert previous_state[0].dataset_state == {0: {"counter": 1}}
    assert previous_state[0].sampler_state["sampler"] == {"counter": 1}
    # TODO: Resolve the previous `sampler_state` associated to `worker_id: 0`.
    worker_id = 1 if num_workers else 0
    assert state[worker_id].sampler_state["sampler"] == {"counter": 2}

    # each worker has its own copy of the dataset
    assert state[0].dataset_state == ({0: {"counter": 2}} if num_workers == 0 else {0: {"counter": 1}})
    target_previous_state = deepcopy(state)

    next(data_fetcher_iter)
    latest_worker_id = data_fetcher.dataloader_iter.state.latest_worker_id
    assert latest_worker_id == 0
    previous_state = data_fetcher.dataloader_iter.previous_state.state
    state = data_fetcher.dataloader_iter.state.state

    assert target_previous_state == previous_state
    assert state[0].sampler_state["sampler"] == {"counter": 3}
    assert state[0].dataset_state == ({0: {"counter": 3}} if num_workers == 0 else {0: {"counter": 2}})

    _teardown_dataloader_get_iterators()
    assert not hasattr(DataLoader, "_ori_get_iterator")
    assert DataLoader._get_iterator == _get_iterator_fn

    _reload_dataloader_state_dict(dataloader, asdict(reloaded_state))
    assert dataloader.sampler.counter == dataloader.dataset.counter == 1
    data_fetcher.teardown()


class RandomFaultTolerantDataset(RandomGetItemDataset):
    def __init__(self, *args, seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self._cache_state_dict = None
        self.generator = None
        self.counter_debug = 0

    @property
    def worker_id(self):
        info = get_worker_info()
        return info.id if info else 0

    def __getitem__(self, index):
        if self._cache_state_dict:
            state_dict = self._cache_state_dict[self.worker_id]
            self.generator = random.Random()
            self.generator.setstate(state_dict["random_state"])
            self._cache_state_dict = None

        if not self.generator:
            self.generator = random.Random(self.seed + self.worker_id)
        return torch.tensor(index + self.generator.random())

    def state_dict(self):
        return {self.worker_id: {"random_state": self.generator.getstate()}}

    def load_state_dict(self, state_dict):
        self._cache_state_dict = state_dict


class RandomFaultTolerantSampler(RandomSampler):
    def __init__(self, *args, seed: int = 0, **kwargs):
        generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        self.restarting = True

    def __len__(self):
        return len(self.data_source) - self.counter

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


@pytest.mark.parametrize(
    ["train_dataset_cls", "val_dataset_cls"],
    [
        ([RandomFaultTolerantDataset, RandomFaultTolerantDataset], [RandomFaultTolerantDataset]),
    ],
)
@pytest.mark.parametrize("val_check_interval", [0.5])
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "2"})
def test_fault_tolerant_manual_mode(val_check_interval, train_dataset_cls, val_dataset_cls, tmpdir):
    class TestModel(BoringModel):
        def __init__(self, should_fail: bool = False):
            super().__init__()
            self.layer = torch.nn.Linear(1, 2)
            self.should_fail = should_fail
            self.batches = []

        def training_step(self, batch, batch_idx):
            if self.should_fail and batch_idx == 7:
                raise CustomException
            self.batches.append(batch)
            losses = []
            for b in batch:
                losses.append(super().training_step(b, batch_idx)["loss"])
            return torch.stack(losses).mean()

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            pass

        validation_epoch_end = None

        def _create_dataloader_kwargs(self, dataset_class, dataset_len, seed, num_workers):
            dl_kwargs = {}
            dl_kwargs["dataset"] = dataset_class(dataset_len, 1, seed=seed)
            dl_kwargs["sampler"] = RandomFaultTolerantSampler(dl_kwargs["dataset"], seed=seed)
            dl_kwargs["num_workers"] = num_workers
            dl_kwargs["batch_size"] = 1
            return dl_kwargs

        def train_dataloader(self):
            return [
                DataLoader(
                    **self._create_dataloader_kwargs(
                        dataset_class, 10, seed, seed + 1 if val_check_interval == 1.0 else 0
                    )
                )
                for seed, dataset_class in enumerate(train_dataset_cls)
            ]

        def val_dataloader(self):
            return [
                DataLoader(**self._create_dataloader_kwargs(dataset_class, 1, seed, 0))
                for seed, dataset_class in enumerate(val_dataset_cls)
            ]

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    seed_everything(42)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval)
    trainer.fit(model)
    total_batches = model.batches
    total_weight = deepcopy(model.layer.weight)
    trainer.train_dataloader = None

    seed_everything(42)
    model = TestModel(should_fail=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval)
    with pytest.raises(CustomException):
        trainer.fit(model)
    trainer.train_dataloader = None
    failed_batches = model.batches
    failed_weight = deepcopy(model.layer.weight)

    checkpoint_path = str(tmpdir / ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path)

    seed_everything(42)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=val_check_interval)
    trainer.fit(model, ckpt_path=checkpoint_path)
    trainer.train_dataloader = None
    restart_batches = model.batches

    torch.testing.assert_allclose(total_batches, failed_batches + restart_batches)
    assert not torch.equal(total_weight, failed_weight)
    assert torch.equal(total_weight, model.layer.weight)
