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
from collections.abc import Iterable
from unittest import mock

import pytest
import torch
from torch.optim import Adam
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data._utils.worker import get_worker_info
from torch.utils.data.dataloader import _InfiniteConstantSampler, DataLoader
from torch.utils.data.dataset import IterableDataset, Dataset

from pytorch_lightning import Callback, seed_everything, Trainer, LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.overrides.distributed import (
    CaptureIterativeDataset,
    FastForwardSampler,
    IndexBatchSamplerWrapper,
    UnrepeatedDistributedSampler,
)
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.enums import BatchKeys
from tests.helpers.boring_model import BoringModel, RandomDataset


@pytest.mark.parametrize("shuffle", [False, True])
def test_unrepeated_distributed_sampler(shuffle, tmpdir):
    """Test each rank will receive a different number of elements."""

    seed_everything(42)
    world_size = 4
    samplers = []
    dataset = range(103)
    for rank in range(world_size):
        samplers.append(UnrepeatedDistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle))

    indices = [list(s) for s in samplers]
    assert len(indices[0]) == 26
    assert len(indices[1]) == 26
    assert len(indices[2]) == 26
    assert len(indices[3]) == 25

    assert indices[0][-1] == 18 if shuffle else 100
    assert indices[1][-1] == 30 if shuffle else 101
    assert indices[2][-1] == 29 if shuffle else 102
    assert indices[3][-1] == 35 if shuffle else 99


def test_index_batch_sampler(tmpdir):
    """Test `IndexBatchSampler` properly extracts indices."""
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

    assert batch_sampler.batch_size == index_batch_sampler.batch_size
    assert batch_sampler.drop_last == index_batch_sampler.drop_last
    assert batch_sampler.sampler is sampler

    for batch in index_batch_sampler:
        assert index_batch_sampler.batch_indices == batch


def test_index_batch_sampler_methods():
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

    assert isinstance(index_batch_sampler, Iterable)
    assert has_len(index_batch_sampler)


def test_fast_forward_on_batch_sampler():
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)
    index_batch_sampler.setup(1, 1, False)

    assert isinstance(index_batch_sampler, Iterable)
    assert has_len(index_batch_sampler)

    index_batch_sampler_iter = iter(index_batch_sampler)

    assert next(index_batch_sampler_iter) == [0, 1, 2]
    assert next(index_batch_sampler_iter) == [3, 4, 5]

    state_dict = index_batch_sampler.state_dict(2)

    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)
    index_batch_sampler.setup(1, 1, False)
    index_batch_sampler.load_state_dict(state_dict)

    index_batch_sampler_iter = iter(index_batch_sampler)
    assert next(index_batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_sequential_sampler():
    dataset = range(15)
    sampler = FastForwardSampler(SequentialSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [0, 1, 2]
    assert next(batch_sampler_iter) == [3, 4, 5]

    state_dict = sampler.state_dict(2)
    assert state_dict["current_iteration"] == 6

    dataset = range(15)
    sampler = FastForwardSampler(SequentialSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)
    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_random_sampler():
    seed_everything(42)

    dataset = range(15)
    sampler = FastForwardSampler(RandomSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [14, 9, 1]
    assert next(batch_sampler_iter) == [7, 11, 3]
    assert next(batch_sampler_iter) == [12, 8, 2]

    state_dict = sampler.state_dict(3)
    assert state_dict["current_iteration"] == 9
    state_dict["current_iteration"] = 6

    dataset = range(15)
    sampler = FastForwardSampler(RandomSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)
    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == [12, 8, 2]
    has_raised = False
    try:
        for _ in range(5):
            next(batch_sampler_iter)
    except StopIteration:
        has_raised = True
        assert sampler.rng_state is None
        assert sampler.current_iteration == 0
        sampler.load_state_dict(sampler.state_dict(0))
    assert has_raised


def test_fast_forward_sampler_replacement(tmpdir):

    seed_everything(42)

    class CustomBatchSampler(BatchSampler):
        pass

    class CheckFastForwardSamplerInjection(Callback):

        def __init__(self):
            self.has_called = False

        def on_train_batch_end(self, trainer, *args) -> None:
            sampler = trainer.train_dataloader.loaders.fast_forward_sampler
            wrapping_batch_sampler = isinstance(sampler.sampler, BatchSampler)

            num_batches = 2

            if wrapping_batch_sampler:
                assert isinstance(sampler, FastForwardSampler)
                current_iteration = num_batches
            else:
                assert isinstance(sampler, FastForwardSampler)
                current_iteration = 2 * 3

            if trainer.fit_loop.batch_idx == 1:
                assert sampler.state_dict(num_batches)["current_iteration"] == current_iteration
                self.has_called = True

    dataset = RandomDataset(32, 64)
    train_dataloader = DataLoader(dataset, batch_size=3, num_workers=1)
    trainer_kwargs = dict(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, num_sanity_val_steps=0, limit_val_batches=0
    )
    model = BoringModel()
    callback = CheckFastForwardSamplerInjection()
    trainer = Trainer(**trainer_kwargs, callbacks=callback)
    trainer.fit(model, train_dataloader=train_dataloader)

    train_dataloader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(batch_size=8, sampler=SequentialSampler(dataset), drop_last=True),
        num_workers=2,
    )

    trainer = Trainer(**trainer_kwargs, callbacks=callback)
    trainer.fit(model, train_dataloader=train_dataloader)


class RangeIterativeDataset(IterableDataset):

    def __init__(self, data, num_workers: int, batch_size: int, is_in_workers: bool, state_dict=None):
        self.data = list(data)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_in_workers = is_in_workers
        self.state_dict = state_dict

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info and self.num_workers == 2:
            id = worker_info.id
            num_samples = len(self.data)
            if id == 0:
                self.data = list(self.data)[:num_samples // 2]
            else:
                self.data = list(self.data)[num_samples // 2:]
            self.user_sampler = RandomSampler(self.data)
        else:
            self.user_sampler = RandomSampler(self.data)

        sampler = FastForwardSampler(self.user_sampler)
        sampler.setup(self.batch_size, self.num_workers, self.is_in_workers)
        if self.state_dict is not None:
            sampler.load_state_dict(self.state_dict[0]["iter_sampler"])
            self.state_dict = None
        self.sampler = sampler
        self.iter_sampler = iter(self.sampler)
        return self

    def __next__(self):
        return self.data[next(self.iter_sampler)]


@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [3])
def test_fast_forward_sampler_over_iterative_dataset(num_workers, batch_size, tmpdir):

    initial_seed = seed_everything(42)
    generator = torch.Generator()
    generator.manual_seed(initial_seed)
    dataset = RangeIterativeDataset(range(20), num_workers, batch_size, True)
    dataset = CaptureIterativeDataset(dataset, num_workers, batch_size, True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    iter_dataloader = iter(dataloader)
    batches = []
    for _ in range(5):
        batches.append(next(iter_dataloader))

    if num_workers == 0:
        batch_0_expected = torch.tensor([4, 3, 12])
        batch_1_expected = torch.tensor([18, 0, 7])
        batch_2_expected = torch.tensor([8, 14, 11])
        batch_3_expected = torch.tensor([5, 10, 13])
        batch_4_expected = torch.tensor([17, 19, 15])
    elif num_workers == 1:
        batch_0_expected = torch.tensor([3, 18, 17])
        batch_1_expected = torch.tensor([13, 2, 19])
        batch_2_expected = torch.tensor([6, 4, 7])
        batch_3_expected = torch.tensor([1, 14, 5])
        batch_4_expected = torch.tensor([12, 8, 16])
    else:
        batch_0_expected = torch.tensor([3, 4, 5])
        batch_1_expected = torch.tensor([10, 12, 14])
        batch_2_expected = torch.tensor([7, 0, 9])
        batch_3_expected = torch.tensor([16, 18, 17])
        batch_4_expected = torch.tensor([8, 1, 2])

    assert torch.equal(batches[0]["data"], batch_0_expected)
    assert torch.equal(batches[1]["data"], batch_1_expected)
    assert torch.equal(batches[2]["data"], batch_2_expected)
    assert torch.equal(batches[3]["data"], batch_3_expected)
    assert torch.equal(batches[4]["data"], batch_4_expected)

    # restarting on batch_1 and getting 3 extra batches

    def parse_metadata(batch):
        return {
            k: {
                batch[BatchKeys.PL_SAMPLERS]["id"][-1].item(): {
                    "current_iteration": v["current_iteration"][-1].item(),
                    "rng_state": v["rng_state"][-1]
                }
            }
            for k, v in batch[BatchKeys.PL_SAMPLERS].items() if k != "id"
        }

    state_dict = {0: {'iter_sampler': {}}}
    for batch in batches[:2]:
        metadata = parse_metadata(batch)
        for k, v in metadata.items():
            state_dict[0][k].update(v)

    if num_workers == 2:
        assert len(state_dict[0]["iter_sampler"]) == 2

    initial_seed = seed_everything(42)
    generator.manual_seed(initial_seed)
    dataset = RangeIterativeDataset(range(20), num_workers, batch_size, True, state_dict=state_dict)
    dataset = CaptureIterativeDataset(dataset, num_workers, batch_size, True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    iter_dataloader = iter(dataloader)
    batches = []
    for _ in range(3):
        batches.append(next(iter_dataloader))

    assert torch.equal(batches[0]["data"], batch_2_expected)
    assert torch.equal(batches[1]["data"], batch_3_expected)
    assert torch.equal(batches[2]["data"], batch_4_expected)


class CustomIterativeDataset(IterableDataset):

    def __init__(self, dataset, num_workers: int, drop_last: bool = True):
        self.dataset = list(dataset)
        self.num_workers = num_workers
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_workers != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_workers) / self.num_workers)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_workers)

        self.total_size = self.num_samples * self.num_workers

    @property
    def rank(self) -> int:
        info = get_worker_info()
        return info.id if info else 0

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_workers]
        assert len(indices) == self.num_samples

        self.indices = indices
        self.sampler = RandomSampler(indices)
        self.iter_sampler = iter(self.sampler)

        return self

    def __next__(self):
        return self.indices[next(self.iter_sampler)]


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_fast_forward_sampler_iterative_dataset(tmpdir):

    seed_everything(42)

    class CustomException(Exception):
        pass

    class CheckFastForwardSamplerInjection(Callback):

        def __init__(self):
            self.has_called = False
            self.restarting = False

        def _validate_map_dl_idx_sampler_states(self, trainer, num_dataloaders, worker_iterations):
            map_dl_idx_sampler_states = trainer.fit_loop.epoch_loop._map_dl_idx_sampler_states
            assert len(map_dl_idx_sampler_states) == num_dataloaders
            assert len(map_dl_idx_sampler_states[0]["iter_sampler"]) == len([i for i in worker_iterations if i > 0])
            if len(worker_iterations) == 1 and worker_iterations[0] > 0:
                assert map_dl_idx_sampler_states[0]["iter_sampler"][0]["current_iteration"] == worker_iterations[0]
            if len(worker_iterations) == 2 and worker_iterations[1] > 0:
                assert map_dl_idx_sampler_states[0]["iter_sampler"][1]["current_iteration"] == worker_iterations[1]
            if len(worker_iterations) == 3 and worker_iterations[2] > 0:
                assert map_dl_idx_sampler_states[0]["iter_sampler"][2]["current_iteration"] == worker_iterations[2]
            if len(worker_iterations) == 4 and worker_iterations[3] > 0:
                assert map_dl_idx_sampler_states[0]["iter_sampler"][3]["current_iteration"] == worker_iterations[3]

        def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
        ) -> None:
            assert isinstance(trainer.train_dataloader.loaders.sampler, _InfiniteConstantSampler)
            assert isinstance(trainer.train_dataloader.loaders.dataset, CaptureIterativeDataset)
            assert trainer.train_dataloader.loaders.generator.initial_seed() == 42
            assert trainer.train_dataloader.loaders.dataset.initial_seed == 42
            if not self.restarting:
                if trainer.fit_loop.batch_idx == 0:
                    t = torch.tensor([20, 16, 24])
                    self._validate_map_dl_idx_sampler_states(trainer, 1, [3])
                    assert torch.equal(batch, t)
                    assert torch.equal(t % 4, torch.tensor([0, 0, 0]))
                elif trainer.fit_loop.batch_idx == 1:
                    t = torch.tensor([1, 9, 5])
                    self._validate_map_dl_idx_sampler_states(trainer, 1, [3, 3])
                    assert torch.equal(batch, t)
                    assert torch.equal(t % 4, torch.tensor([1, 1, 1]))
                    raise CustomException
            else:
                if trainer.fit_loop.batch_idx == 2:
                    t = torch.tensor([2, 14, 22])
                    self._validate_map_dl_idx_sampler_states(trainer, 1, [0, 0, 3])
                    assert torch.equal(batch, t)
                    assert torch.equal(t % 4, torch.tensor([2, 2, 2]))
                elif trainer.fit_loop.batch_idx == 3:
                    t = torch.tensor([7, 11, 15])
                    self._validate_map_dl_idx_sampler_states(trainer, 1, [0, 0, 3, 3])
                    assert torch.equal(batch, t)
                    assert torch.equal(t % 4, torch.tensor([3, 3, 3]))
                elif trainer.fit_loop.batch_idx == 4:
                    t = torch.tensor([8, 4, 0])
                    self._validate_map_dl_idx_sampler_states(trainer, 1, [6, 0, 3, 3])
                    assert torch.equal(batch, t)
                    assert torch.equal(t % 4, torch.tensor([0, 0, 0]))

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            pass

    model = TestModel()
    model.training_epoch_end = None

    num_workers = 4

    dataset = CustomIterativeDataset(range(30), num_workers)
    train_dataloader = DataLoader(dataset, batch_size=3, num_workers=num_workers)
    trainer_kwargs = dict(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=10, num_sanity_val_steps=0, limit_val_batches=0
    )
    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    cb = CheckFastForwardSamplerInjection()
    callbacks = [cb, ck]
    trainer = Trainer(**trainer_kwargs, callbacks=callbacks)
    try:
        trainer.fit(model, train_dataloader=train_dataloader)
    except CustomException:
        pass

    cb.restarting = True

    dataset = CustomIterativeDataset(range(30), num_workers)
    train_dataloader = DataLoader(dataset, batch_size=3, num_workers=num_workers)
    trainer = Trainer(**trainer_kwargs, resume_from_checkpoint=ck.last_model_path, callbacks=callbacks)
    trainer.fit(model, train_dataloader=train_dataloader)


class MonotonicRandomDataset(Dataset):

    def __getitem__(self, index):
        # 0.{random digits}
        # 1.{random digits}
        # 2.{random digits}
        # ...
        return torch.rand(1) + index

    def __len__(self):
        return 64


class RandomLightningModule(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)
        self.recorded_samples = []

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # print(batch_idx, batch)
        self.recorded_samples.append(batch)
        return {"loss": self(batch).sum()}

    def train_dataloader(self):
        dataset = MonotonicRandomDataset()
        dataloader = DataLoader(dataset, batch_size=2)
        return dataloader

    def configure_optimizers(self):
        return Adam(self.parameters())


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_fastforward_sampler_and_dataset(tmpdir):
    print("initial training")
    seed_everything(1)
    model = RandomLightningModule()
    trainer = Trainer(max_steps=3, progress_bar_refresh_rate=0, weights_summary=None)
    trainer.fit(model)

    print(torch.cat(model.recorded_samples))
    indices = [int(x) for x in torch.cat(model.recorded_samples).floor()]
    assert indices == [0, 1, 2, 3, 4, 5]

    ckpt_file = os.path.join(tmpdir, "one.ckpt")
    trainer.save_checkpoint(ckpt_file)

    print("resuming")
    seed_everything(1)
    model = RandomLightningModule()
    trainer = Trainer(max_steps=6, progress_bar_refresh_rate=0, weights_summary=None, resume_from_checkpoint=ckpt_file)
    trainer.fit(model)

    print(torch.cat(model.recorded_samples))
    indices = [int(x) for x in torch.cat(model.recorded_samples).floor()]
    assert indices == [6, 7, 8, 9]

