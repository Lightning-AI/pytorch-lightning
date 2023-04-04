# Copyright The Lightning AI team.
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
from typing import Any, Iterator

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loops.fetchers import _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tests_pytorch.helpers.runif import RunIf


class IterDataset(IterableDataset):
    def __iter__(self):
        yield 1
        yield 2
        yield 3


class SizedDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return idx + 1


@pytest.mark.parametrize("use_combined_loader", [False, True])
@pytest.mark.parametrize("dataset_cls", [IterDataset, SizedDataset])
@pytest.mark.parametrize("prefetch_batches", list(range(5)))
def test_prefetch_iterator(use_combined_loader, dataset_cls, prefetch_batches):
    fetcher = _PrefetchDataFetcher(prefetch_batches=prefetch_batches)
    assert fetcher.prefetch_batches == prefetch_batches

    if use_combined_loader:
        loader = CombinedLoader([DataLoader(dataset_cls()), DataLoader(dataset_cls())])
    else:
        loader = DataLoader(dataset_cls())
    fetcher.setup(loader)

    def generate():
        generated = [(fetcher.fetched, data, fetcher.done) for data in fetcher]
        assert fetcher.fetched == 3
        assert fetcher.done
        return generated

    # we can only know the last batch with sized iterables or when we prefetch
    is_last_batch = [False, False, prefetch_batches > 0 or dataset_cls is SizedDataset]
    fetched = (
        [1, 2, 3] if dataset_cls is SizedDataset else [1, 2, 3, 3, 3, 3, 3][prefetch_batches : prefetch_batches + 3]
    )
    batches = [[1, 1], [2, 2], [3, 3]] if use_combined_loader else [1, 2, 3]
    expected = list(zip(fetched, batches, is_last_batch))
    assert len(expected) == 3

    assert generate() == expected
    # validate reset works properly.
    assert generate() == expected
    assert fetcher.fetched == 3


@pytest.mark.parametrize("use_combined_loader", [False, True])
def test_profiler_closing(use_combined_loader):
    """Tests if the profiler terminates upon raising a StopIteration on an iterable dataset."""

    class TestDataset(IterableDataset):
        def __init__(self):
            self.list = list(range(1))

        def __iter__(self):
            return iter(self.list)

    fetcher = _PrefetchDataFetcher()
    if use_combined_loader:
        loader = CombinedLoader([DataLoader(TestDataset()), DataLoader(TestDataset())])
    else:
        loader = DataLoader(TestDataset())
    fetcher.setup(loader)
    profiler = SimpleProfiler()
    fetcher._start_profiler = lambda: profiler.start("test")
    fetcher._stop_profiler = lambda: profiler.stop("test")
    iter(fetcher)  # on epoch 0 start
    next(fetcher)  # raises StopIteration exception
    assert not bool(profiler.current_actions)


class EmptyIterDataset(IterableDataset):
    def __iter__(self):
        return iter([])


class EmptySizedDataset(Dataset):
    def __len__(self):
        return 0


@pytest.mark.parametrize("dataset_cls", [EmptyIterDataset, EmptySizedDataset])
@pytest.mark.parametrize("prefetch_batches", list(range(2)))
def test_empty_prefetch_iterator(dataset_cls, prefetch_batches):
    loader = DataLoader(dataset_cls())
    fetcher = _PrefetchDataFetcher(prefetch_batches=prefetch_batches)
    fetcher.setup(loader)

    assert not fetcher.done
    assert not list(fetcher)
    assert fetcher.done


def get_cycles_per_ms() -> float:
    """Get 10 values and remove the 2 max and 2 min and return the avg.

    This is to avoid system disturbance that skew the results, e.g. the very first cuda call likely does a bunch of
    init, which takes much longer than subsequent calls.
    """

    def measure() -> float:
        """Measure and return approximate number of cycles per millisecond for `torch.cuda._sleep` Copied from:

        https://github.com/pytorch/pytorch/blob/v1.9.0/test/test_cuda.py#L81.
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    stats = vals[2 : num - 2]
    return sum(stats) / len(stats)


BATCH_SIZE = 32
DATASET_LEN = 64


@pytest.mark.parametrize("automatic_optimization", [False, True])
def test_fetching_dataloader_iter_opt(automatic_optimization, tmpdir):
    class TestModel(BoringModel):
        def __init__(self, *args, automatic_optimization: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = automatic_optimization
            self.count = 0
            self.batches = []

        def training_step(self, dataloader_iter, batch_idx):
            assert self.count == batch_idx
            assert isinstance(self.trainer.fit_loop._data_fetcher, _DataLoaderIterDataFetcher)
            # fetch 2 batches
            self.batches.append(next(dataloader_iter))
            self.batches.append(next(dataloader_iter))

            batch = self.batches.pop(0)
            assert isinstance(batch, Tensor) or batch is None
            self.count += 2
            if self.automatic_optimization:
                loss = super().training_step(batch, 0)
                with pytest.raises(MisconfigurationException, match="dataloader_iter"):
                    self.log("train_loss", loss["loss"])
                self.log("train_loss", loss["loss"], batch_size=1)
            else:
                opt = self.optimizers()
                loss = self.step(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        def on_train_epoch_end(self):
            assert self.trainer.fit_loop.epoch_loop.batch_progress.current.ready == 33
            assert self.trainer.fit_loop._data_fetcher.fetched == 64
            assert self.count == 64

    model = TestModel(automatic_optimization=automatic_optimization)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)


@pytest.mark.parametrize("fn", ("validate", "test", "predict"))
def test_fetching_dataloader_iter_running_stages(fn, tmp_path):
    class TestModel(BoringModel):
        def fetch(self, data_fetcher, dataloader_iter, batch_idx):
            assert isinstance(data_fetcher, _DataLoaderIterDataFetcher)
            assert data_fetcher.fetched == batch_idx
            batch = next(dataloader_iter)
            assert data_fetcher.fetched == batch_idx + 1
            return batch

        def validation_step(self, dataloader_iter, batch_idx):
            data_fetcher = self.trainer.validate_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter, batch_idx)
            return super().validation_step(batch, batch_idx)

        def test_step(self, dataloader_iter, batch_idx):
            data_fetcher = self.trainer.test_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter, batch_idx)
            return super().test_step(batch, batch_idx)

        def predict_step(self, dataloader_iter, batch_idx):
            data_fetcher = self.trainer.predict_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter, batch_idx)
            return super().test_step(batch, batch_idx)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    trainer_fn = getattr(trainer, fn)
    trainer_fn(model)


@pytest.mark.parametrize("fn", ("validate", "test", "predict"))
def test_fetching_dataloader_iter_running_stages_multiple_dataloaders(fn, tmp_path):
    class MyModel(BoringModel):
        def validation_step(self, dataloader_iter, batch_idx, dataloader_idx):
            ...

        def test_step(self, dataloader_iter, batch_idx, dataloader_idx):
            ...

        def predict_step(self, dataloader_iter, batch_idx, dataloader_idx):
            ...

    def dataloaders():
        return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    model = MyModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    trainer_fn = getattr(trainer, fn)
    with pytest.raises(NotImplementedError, match="dataloader_iter.*is not supported with multiple dataloaders"):
        trainer_fn(model, dataloaders())


class DummyWaitable:
    def __init__(self, val: Any) -> None:
        self.val = val

    def wait(self) -> Any:
        return self.val


class AsyncBoringModel(BoringModel):
    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.batch_i_handle = None
        self.num_batches_processed = 0

    def _async_op(self, batch: Any) -> DummyWaitable:
        return DummyWaitable(val=batch)

    def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
        if self.batch_i_handle is None:
            batch_i_raw = next(dataloader_iter)
            self.batch_i_handle = self._async_op(batch_i_raw)

        # Invariant: _async_op for batch[i] has been initiated
        batch_ip1_handle = None
        is_last = False
        try:
            batch_ip1_raw = next(dataloader_iter)
            batch_ip1_handle = self._async_op(batch_ip1_raw)
        except StopIteration:
            is_last = True

        batch_i = self.batch_i_handle.wait()

        loss = self.step(batch_i)
        loss.backward()
        self.optimizers().step()
        self.optimizers().zero_grad()

        self.batch_i_handle = batch_ip1_handle
        self.num_batches_processed += 1

        return {"loss": loss, "is_last": is_last}

    def train_dataloader(self):
        return DataLoader(RandomDataset(BATCH_SIZE, DATASET_LEN))


def test_training_step_with_dataloader_access(tmpdir) -> None:
    """A baseline functional test for `training_step` with dataloader access."""
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = AsyncBoringModel()
    trainer.fit(m)
    assert m.num_batches_processed == DATASET_LEN, f"Expect all {DATASET_LEN} batches to be processed."


@pytest.mark.parametrize("trigger_stop_iteration", [False, True])
def test_stop_iteration(trigger_stop_iteration, tmpdir):
    """Verify that StopIteration properly terminates the training when this is triggered from the current
    `dataloader_iter`"""
    EXPECT_NUM_BATCHES_PROCESSED = 2

    class TestModel(AsyncBoringModel):
        def __init__(self, trigger_stop_iteration) -> None:
            super().__init__()
            self.trigger_stop_iteration = trigger_stop_iteration

        def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
            output = super().training_step(dataloader_iter)
            batch_idx = self.trainer.fit_loop.epoch_loop.batch_idx
            if self.trigger_stop_iteration and batch_idx == EXPECT_NUM_BATCHES_PROCESSED:
                raise StopIteration
            return output

        def train_dataloader(self):
            if self.trigger_stop_iteration:
                return DataLoader(RandomDataset(BATCH_SIZE, 2 * EXPECT_NUM_BATCHES_PROCESSED))
            return DataLoader(RandomDataset(BATCH_SIZE, EXPECT_NUM_BATCHES_PROCESSED))

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = TestModel(trigger_stop_iteration)
    trainer.fit(m)
    expected = EXPECT_NUM_BATCHES_PROCESSED
    if trigger_stop_iteration:
        expected *= 2
    assert m.num_batches_processed == expected


def test_on_train_batch_start_overridden(tmpdir) -> None:
    """Verify that a `MisconfigurationException` is raised when `on_train_batch_start` is overridden on the
    `LightningModule`."""

    class InvalidModel(AsyncBoringModel):
        def on_train_batch_start(self, batch, batch_idx):
            pass

    trainer = Trainer(fast_dev_run=1, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.warns(match="InvalidModel.on_train_batch_start` hook may not match"):
        trainer.fit(m)


def test_on_train_batch_end_overridden(tmpdir) -> None:
    """Verify that a `MisconfigurationException` is raised when `on_train_batch_end` is overridden on the
    `LightningModule`."""

    class InvalidModel(AsyncBoringModel):
        def on_train_batch_end(self, *_):
            pass

    trainer = Trainer(fast_dev_run=1, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.warns(match="InvalidModel.on_train_batch_end` hook may not match"):
        trainer.fit(m)


def test_transfer_hooks_with_unpacking(tmpdir):
    """This test asserts the `transfer_batch` hooks are called only once per batch."""

    class RandomDictDataset(RandomDataset):
        def __getitem__(self, index):
            return {"x": self.data[index], "y_true": torch.ones((2,)), "other": torch.ones((1,))}

    class BoringDataModule(LightningDataModule):
        count_called_on_before_batch_transfer = 0
        count_called_transfer_batch_to_device = 0
        count_called_on_after_batch_transfer = 0

        def train_dataloader(self):
            return DataLoader(RandomDictDataset(32, 2))

        def val_dataloader(self):
            return DataLoader(RandomDictDataset(32, 2))

        def on_before_batch_transfer(self, batch, dataloader_idx: int):
            self.count_called_on_before_batch_transfer += 1
            return batch["x"], batch["y_true"]

        def transfer_batch_to_device(self, *args, **kwargs):
            self.count_called_transfer_batch_to_device += 1
            return super().transfer_batch_to_device(*args, **kwargs)

        def on_after_batch_transfer(self, batch, dataloader_idx: int):
            self.count_called_on_after_batch_transfer += 1
            return super().on_after_batch_transfer(batch, dataloader_idx)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            x, _ = batch
            return super().training_step(x, batch_idx)

        def validation_step(self, batch, batch_idx):
            x, _ = batch
            return super().validation_step(x, batch_idx)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, num_sanity_val_steps=0)
    dm = BoringDataModule()
    trainer.fit(TestModel(), datamodule=dm)
    assert dm.count_called_on_before_batch_transfer == 4
    assert dm.count_called_transfer_batch_to_device == 4
    assert dm.count_called_on_after_batch_transfer == 4


@RunIf(skip_windows=True)  # TODO: all durations are 0 on Windows
def test_fetching_is_profiled():
    """Test that fetching is profiled."""

    class MyModel(BoringModel):
        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            return [super().val_dataloader(), super().val_dataloader()]

    model = MyModel()
    fast_dev_run = 2
    trainer = Trainer(
        fast_dev_run=fast_dev_run,
        profiler="simple",
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)

    profiler = trainer.profiler
    assert isinstance(profiler, SimpleProfiler)

    # validation
    key = "[_EvaluationLoop].val_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    # +1 because we fetch one extra batch before breaking the loop when the fast_dev_run condition allows
    assert len(durations) == 2 * fast_dev_run + 1
    assert all(d > 0 for d in durations)
    # training
    key = "[_TrainingEpochLoop].train_dataloader_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run
    assert all(d > 0 for d in durations)
    # test
    key = "[_EvaluationLoop].test_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run + 1
    assert all(d > 0 for d in durations)
    # predict
    key = "[_PredictionLoop].predict_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run + 1
    assert all(d > 0 for d in durations)

    # now test profiling when the dataloader_iter is polled manually
    class MyModel(BoringModel):
        def training_step(self, dataloader_iter):
            _ = next(dataloader_iter)
            batch = next(dataloader_iter)
            return super().training_step(batch, 0)

    model = MyModel()
    trainer = Trainer(
        fast_dev_run=1,
        profiler="simple",
        limit_val_batches=0,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model)

    profiler = trainer.profiler
    assert isinstance(profiler, SimpleProfiler)

    key = "[_TrainingEpochLoop].train_dataloader_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == 2  # 2 polls in training_step
    assert all(d > 0 for d in durations)
