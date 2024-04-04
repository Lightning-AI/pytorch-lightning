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
from collections import Counter
from typing import Any, Iterator

import pytest
import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loops.fetchers import _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tests_pytorch.helpers.runif import RunIf


class IterDataset(IterableDataset):
    def __init__(self, size=3):
        self.size = size

    def __iter__(self):
        yield from range(1, self.size + 1)


class SizedDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return idx + 1


@pytest.mark.parametrize("multiple_iterables", [False, True])
@pytest.mark.parametrize("dataset_cls", [IterDataset, SizedDataset])
@pytest.mark.parametrize("prefetch_batches", list(range(5)))
def test_prefetch_iterator(multiple_iterables, dataset_cls, prefetch_batches):
    fetcher = _PrefetchDataFetcher(prefetch_batches=prefetch_batches)
    assert fetcher.prefetch_batches == prefetch_batches

    if multiple_iterables:
        loader = CombinedLoader([DataLoader(dataset_cls()), DataLoader(dataset_cls())])
    else:
        loader = CombinedLoader(DataLoader(dataset_cls()))
    fetcher.setup(loader)

    def generate():
        generated = [(fetcher.fetched, data, fetcher.done) for data, batch_idx, dataloader_idx in fetcher]
        assert fetcher.fetched == 3
        assert fetcher.done
        return generated

    # we can only know the last batch with sized iterables or when we prefetch
    is_last_batch = [False, False, prefetch_batches > 0 or dataset_cls is SizedDataset]
    fetched = (
        [1, 2, 3] if dataset_cls is SizedDataset else [1, 2, 3, 3, 3, 3, 3][prefetch_batches : prefetch_batches + 3]
    )
    batches = [[1, 1], [2, 2], [3, 3]] if multiple_iterables else [1, 2, 3]
    expected = list(zip(fetched, batches, is_last_batch))
    assert len(expected) == 3

    assert generate() == expected
    # validate reset works properly.
    assert generate() == expected
    assert fetcher.fetched == 3


@pytest.mark.parametrize("multiple_iterables", [False, True])
def test_profiler_closing(multiple_iterables):
    """Tests if the profiler terminates upon raising a StopIteration on an iterable dataset."""

    class TestDataset(IterableDataset):
        def __init__(self):
            self.list = list(range(1))

        def __iter__(self):
            return iter(self.list)

    fetcher = _PrefetchDataFetcher()
    if multiple_iterables:
        loader = CombinedLoader([DataLoader(TestDataset()), DataLoader(TestDataset())])
    else:
        loader = CombinedLoader(TestDataset())
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
@pytest.mark.parametrize("prefetch_batches", [0, 1])
def test_empty_prefetch_iterator(dataset_cls, prefetch_batches):
    loader = CombinedLoader(DataLoader(dataset_cls()))
    fetcher = _PrefetchDataFetcher(prefetch_batches=prefetch_batches)
    fetcher.setup(loader)
    iter(fetcher)

    if dataset_cls is EmptySizedDataset:
        assert fetcher.done  # for 0 length sized datasets we know we're done already
    else:
        # if we're prefetching, we can know in advance if the dataset is empty
        assert fetcher.done == (prefetch_batches > 0)
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
        # cycles_per_ms
        return 1000000 / start.elapsed_time(end)

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
def test_fetching_dataloader_iter_opt(automatic_optimization, tmp_path):
    class TestModel(BoringModel):
        def __init__(self, *args, automatic_optimization: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = automatic_optimization
            self.count = 0
            self.batches = []

        def training_step(self, dataloader_iter):
            assert isinstance(self.trainer.fit_loop._data_fetcher, _DataLoaderIterDataFetcher)
            # fetch 2 batches
            batch, batch_idx, _ = next(dataloader_iter)
            self.batches.append(batch)
            batch, batch_idx, _ = next(dataloader_iter)
            self.batches.append(batch)

            batch = self.batches.pop(0)
            assert isinstance(batch, Tensor) or batch is None
            self.count = batch_idx + 1
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
            # since the dataset is sized, the loop stops at the limit even though the training_step controls the
            # consumption of batches
            assert self.trainer.fit_loop.epoch_loop.batch_progress.current.ready == 32
            assert self.trainer.fit_loop._data_fetcher.fetched == 64
            assert self.count == 64

    model = TestModel(automatic_optimization=automatic_optimization)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="cpu")
    trainer.fit(model)


@pytest.mark.parametrize("fn", ["validate", "test", "predict"])
def test_fetching_dataloader_iter_running_stages(fn, tmp_path):
    class TestModel(BoringModel):
        def fetch(self, data_fetcher, dataloader_iter):
            assert isinstance(data_fetcher, _DataLoaderIterDataFetcher)
            batch, batch_idx, _ = next(dataloader_iter)
            assert data_fetcher.fetched == batch_idx + 1
            return batch

        def validation_step(self, dataloader_iter):
            data_fetcher = self.trainer.validate_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter)
            return super().validation_step(batch, 0)

        def test_step(self, dataloader_iter):
            data_fetcher = self.trainer.test_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter)
            return super().test_step(batch, 0)

        def predict_step(self, dataloader_iter):
            data_fetcher = self.trainer.predict_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter)
            return super().test_step(batch, 0)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1, accelerator="cpu")
    trainer_fn = getattr(trainer, fn)
    trainer_fn(model)


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
            batch_i_raw, _, _ = next(dataloader_iter)
            self.num_batches_processed += 1
            self.batch_i_handle = self._async_op(batch_i_raw)

        # Invariant: _async_op for batch[i] has been initiated
        batch_ip1_handle = None
        is_last = False
        try:
            batch_ip1_raw, _, _ = next(dataloader_iter)
            self.num_batches_processed += 1
            batch_ip1_handle = self._async_op(batch_ip1_raw)
        except StopIteration:
            is_last = True

        batch_i = self.batch_i_handle.wait()

        loss = self.step(batch_i)
        loss.backward()
        self.optimizers().step()
        self.optimizers().zero_grad()

        self.batch_i_handle = batch_ip1_handle

        return {"loss": loss, "is_last": is_last}

    def train_dataloader(self):
        return DataLoader(RandomDataset(BATCH_SIZE, DATASET_LEN))


def test_training_step_with_dataloader_iter(tmp_path) -> None:
    """A baseline functional test for `training_step` with dataloader access."""
    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, accelerator="cpu")
    m = AsyncBoringModel()
    trainer.fit(m)
    assert m.num_batches_processed == DATASET_LEN, f"Expect all {DATASET_LEN} batches to be processed."


class DataLoaderIterMonitorModel(BoringModel):
    def __init__(self, fetches_per_step):
        super().__init__()
        self.fetches_per_step = fetches_per_step
        self.record = {
            "training": Counter(),
            "validation": Counter(),
            "sanity_validation": Counter(),
            "test": Counter(),
            "predict": Counter(),
        }

    def shared_step(self, dataloader_iter, stage):
        self.record[stage]["entered"] += 1
        for i in range(self.fetches_per_step):
            try:
                batch, _, __ = next(dataloader_iter)
            except StopIteration:
                self.record[stage]["raised"] += 1
                return None
            self.record[stage]["fetched"] += 1
        return self.layer(batch).sum()

    def training_step(self, dataloader_iter):
        return self.shared_step(dataloader_iter, "training")

    def validation_step(self, dataloader_iter):
        stage = "sanity_validation" if self.trainer.sanity_checking else "validation"
        return self.shared_step(dataloader_iter, stage)

    def test_step(self, dataloader_iter):
        return self.shared_step(dataloader_iter, "test")

    def predict_step(self, dataloader_iter):
        return self.shared_step(dataloader_iter, "predict")


@pytest.mark.parametrize(
    ("limit_sanity_val_batches", "limit_train_batches", "limit_eval_batches"),
    [
        (None, None, None),
        (0, 0, 0),
        (2, 2, 2),  # limits are lower than dataloader length
        (100, 100, 100),  # limits are higher than dataloader length
    ],
)
def test_step_methods_with_dataloader_iter(limit_sanity_val_batches, limit_train_batches, limit_eval_batches, tmp_path):
    global_batch_size = 4
    micro_batch_size = 2
    fetches_per_step = global_batch_size // micro_batch_size
    data = DataLoader(RandomDataset(32, length=16), batch_size=micro_batch_size)
    assert len(data) == 8

    limit_sanity_val_batches = 2 if limit_sanity_val_batches is None else limit_sanity_val_batches
    limit_train_batches = limit_train_batches
    limit_val_batches = limit_eval_batches
    limit_test_batches = limit_eval_batches
    limit_predict_batches = limit_eval_batches
    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        limit_predict_batches=limit_predict_batches,
        num_sanity_val_steps=limit_sanity_val_batches,
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, data, data)

    def length(iterable, limit):
        return len(iterable) if limit is None else min(limit, len(data))

    assert model.record["sanity_validation"]["entered"] == length(data, limit_sanity_val_batches) // fetches_per_step
    assert model.record["sanity_validation"]["fetched"] == length(data, limit_sanity_val_batches)
    assert model.record["sanity_validation"]["raised"] == 0
    assert model.record["training"]["entered"] == length(data, limit_train_batches) // fetches_per_step
    assert model.record["training"]["fetched"] == length(data, limit_train_batches)
    assert model.record["training"]["raised"] == 0
    assert model.record["validation"]["entered"] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record["validation"]["fetched"] == length(data, limit_eval_batches)
    assert model.record["validation"]["raised"] == 0

    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer.validate(model, data)
    assert model.record["validation"]["entered"] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record["validation"]["fetched"] == length(data, limit_eval_batches)
    assert model.record["validation"]["raised"] == 0

    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer.test(model, data)
    assert model.record["test"]["entered"] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record["test"]["fetched"] == length(data, limit_eval_batches)
    assert model.record["test"]["raised"] == 0

    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer.predict(model, data)
    assert model.record["predict"]["entered"] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record["predict"]["fetched"] == length(data, limit_eval_batches)
    assert model.record["predict"]["raised"] == 0


@pytest.mark.parametrize("trigger_stop_iteration", [False, True])
def test_stop_iteration_with_dataloader_iter(trigger_stop_iteration, tmp_path):
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

    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, accelerator="cpu")
    m = TestModel(trigger_stop_iteration)
    trainer.fit(m)
    expected = EXPECT_NUM_BATCHES_PROCESSED
    if trigger_stop_iteration:
        expected *= 2
    assert m.num_batches_processed == expected


def test_transfer_hooks_with_unpacking(tmp_path):
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

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, num_sanity_val_steps=0)
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
        accelerator="cpu",
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
    assert len(durations) == 2 * fast_dev_run
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
    assert len(durations) == fast_dev_run
    assert all(d > 0 for d in durations)
    # predict
    key = "[_PredictionLoop].predict_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run
    assert all(d > 0 for d in durations)

    # now test profiling when the dataloader_iter is polled manually
    class MyModel(BoringModel):
        def training_step(self, dataloader_iter):
            _ = next(dataloader_iter)
            batch, _, _ = next(dataloader_iter)
            return super().training_step(batch, 0)

    model = MyModel()
    trainer = Trainer(
        fast_dev_run=2,
        profiler="simple",
        limit_val_batches=0,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
    )
    trainer.fit(model)

    profiler = trainer.profiler
    assert isinstance(profiler, SimpleProfiler)

    key = "[_TrainingEpochLoop].train_dataloader_next"
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == 2  # 2 polls in training_step
    assert all(d > 0 for d in durations)


@pytest.mark.parametrize("iterable", [[1, 2, 3], IterDataset()])
def test_done_dataloader_iter(iterable):
    loader = CombinedLoader(iterable)
    fetcher = _DataLoaderIterDataFetcher()
    fetcher.setup(loader)
    iter(fetcher)

    assert not fetcher.done
    dataloader_iter = next(fetcher)
    for i in range(5):  # doesn't matter how many times you next this, the dataloader_iter needs to be consumed
        assert next(fetcher) is next(fetcher)

    assert not dataloader_iter.done
    assert dataloader_iter.data_fetcher is fetcher

    assert not dataloader_iter.done
    assert next(dataloader_iter)[0] == 1
    assert not dataloader_iter.done
    assert next(dataloader_iter)[0] == 2
    assert not dataloader_iter.done

    assert next(dataloader_iter)[0] == 3
    if isinstance(iterable, list):
        # with sized data, we know we're done
        assert dataloader_iter.done
    else:
        # with unsized data, the StopIteration needs to be raised
        assert not dataloader_iter.done

    with pytest.raises(StopIteration):
        next(dataloader_iter)
    assert dataloader_iter.done


@pytest.mark.parametrize(
    ("mode", "iterables", "limit", "num_fetches", "expected"),
    [
        # sized
        ("min_size", [[1, 2, 3]], None, 2, False),
        ("min_size", [[1, 2, 3]], None, 3, True),
        ("min_size", [[1, 2, 3]], 1, 1, True),
        ("min_size", [[1, 2], [1, 2, 3]], None, 1, False),
        ("min_size", [[1, 2], [1, 2, 3]], None, 2, True),
        ("min_size", [[1, 2], [1, 2, 3]], 1, 1, True),
        ("max_size", [[1, 2], [1, 2, 3]], None, 2, False),
        ("max_size", [[1, 2], [1, 2, 3]], 2, 2, True),
        ("max_size", [[1, 2], [1, 2, 3]], 100, 3, True),  # limit exceeds largest iterable
        ("max_size_cycle", [[1, 2], [1, 2, 3]], None, 2, False),
        ("max_size_cycle", [[1, 2], [1, 2, 3]], 2, 2, True),
        ("max_size_cycle", [[1, 2], [1, 2, 3]], 100, 3, True),  # limit exceeds largest iterable
        ("sequential", [[1, 2], [1, 2, 3]], None, 2, False),
        ("sequential", [[1, 2], [1, 2, 3]], 2, 2, False),
        ("sequential", [[1, 2], [1, 2, 3]], 2, 4, True),  # limit in all iterables needs to be reached
        ("sequential", [[1, 2], [1, 2, 3]], 100, 5, True),  # limit exceeds largest iterable
        # unsized
        ("min_size", [IterDataset()], None, 2, False),
        ("min_size", [IterDataset()], None, 3, False),  # not sized, no prefetching -> can't know if done
        ("min_size", [IterDataset()], 1, 1, True),
        ("min_size", [IterDataset(2), IterDataset(3)], None, 1, False),
        ("min_size", [IterDataset(2), IterDataset(3)], None, 2, False),  # not sized, no prefetching -> can't know
        ("min_size", [IterDataset(2), IterDataset(3)], 1, 1, True),
        ("max_size", [IterDataset(2), IterDataset(3)], None, 2, False),
        ("max_size", [IterDataset(2), IterDataset(3)], 2, 2, True),
        ("max_size", [IterDataset(2), IterDataset(3)], 100, 3, False),  # not sized, no prefetching -> can't know
        ("max_size_cycle", [IterDataset(2), IterDataset(3)], None, 2, False),
        ("max_size_cycle", [IterDataset(2), IterDataset(3)], 2, 2, True),
        ("max_size_cycle", [IterDataset(2), IterDataset(3)], 100, 3, False),  # not sized, no prefetching -> can't know
        ("sequential", [IterDataset(2), IterDataset(3)], None, 2, False),
        ("sequential", [IterDataset(2), IterDataset(3)], 2, 2, False),  # not sized, no prefetching -> can't know
        ("sequential", [IterDataset(2), IterDataset(3)], 2, 4, True),  # limit in all iterables needs to be reached
        ("sequential", [IterDataset(2), IterDataset(3)], 100, 5, False),  # not sized, no prefetching -> can't know
        # sized and unsized mixed
        ("min_size", [[1, 2], IterDataset(3)], None, 1, False),
        ("min_size", [[1, 2], IterDataset(3)], None, 2, True),  # smallest is sized -> done follows the limit
        ("max_size", [IterDataset(2), [1, 2, 3]], None, 2, False),
        ("max_size", [IterDataset(2), [1, 2, 3]], None, 3, False),  # 1st iterable is unsized -> can't know max
        ("max_size_cycle", [IterDataset(2), [1, 2, 3]], None, 2, False),
        ("max_size_cycle", [IterDataset(2), [1, 2, 3]], None, 3, False),
        ("sequential", [[1, 2], IterDataset(3)], 2, 2, False),
        ("sequential", [[1, 2], IterDataset(3)], 2, 4, True),  # limit in all iterables needs to be reached
    ],
)
def test_done_dataloader_iter_with_limit(mode, iterables, limit, num_fetches, expected):
    """Test that the `done` property for `dataloader_iter` gets set as expected."""
    loader = CombinedLoader(iterables, mode=mode)
    fetcher = _DataLoaderIterDataFetcher()
    loader.limits = limit
    fetcher.setup(loader)
    iter(fetcher)

    assert fetcher.done == (limit == 0)
    if num_fetches == 0:
        return

    dataloader_iter = next(fetcher)

    assert not dataloader_iter.done
    for _ in range(num_fetches):
        next(dataloader_iter)
    assert dataloader_iter.done == expected
    assert fetcher.done == expected

    if fetcher.done:
        with pytest.raises(StopIteration):
            next(dataloader_iter)


@pytest.mark.parametrize("mode", ["min_size", "max_size_cycle", "max_size", "sequential"])
def test_done_dataloader_iter_empty_iterables(mode):
    """Test that the `done` property for `dataloader_iter` gets set as expected for empty iterables."""
    fetcher = _DataLoaderIterDataFetcher()

    # single empty iterable
    loader = CombinedLoader([], mode=mode)
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done
    # multiple iterables and all are empty
    loader = CombinedLoader([[], []], mode=mode)
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done
    # one empty, one non-empty
    loader = CombinedLoader([[], [1, 2, 3]], mode=mode)
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done == (mode == "min_size")


@pytest.mark.parametrize("mode", ["min_size", "max_size_cycle", "max_size", "sequential"])
@pytest.mark.parametrize("iterables", [[], [IterDataset()], [[], [1, 2, 3]]])
def test_done_dataloader_iter_zero_limit(iterables, mode):
    """Test that the `done` property for `dataloader_iter` gets set as expected when the limit is 0."""
    fetcher = _DataLoaderIterDataFetcher()
    loader = CombinedLoader(iterables, mode=mode)
    loader.limits = 0
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done
