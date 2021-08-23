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
import os
from time import time
from typing import Any, Iterator, Type
from unittest import mock

import pytest
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import DataFetcher, DataLoaderIterDataFetcher, InterBatchParallelDataFetcher
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("use_combined_loader", [False, True])
def test_prefetch_iterator(use_combined_loader):
    """Test the DataFetcher with PyTorch IterableDataset."""

    class IterDataset(IterableDataset):
        def __iter__(self):
            yield 1
            yield 2
            yield 3

    for prefetch_batches in range(0, 4):
        if use_combined_loader:
            loader = CombinedLoader([DataLoader(IterDataset()), DataLoader(IterDataset())])
            expected = [
                ([tensor([1]), tensor([1])], False),
                ([tensor([2]), tensor([2])], False),
                ([tensor([3]), tensor([3])], True),
            ]
        else:
            loader = DataLoader(IterDataset())
            expected = [(1, False), (2, False), (3, True)]
        iterator = DataFetcher(prefetch_batches=prefetch_batches)
        prefetch_batches += 1
        assert iterator.prefetch_batches == prefetch_batches
        iterator.setup(loader)

        def generate():
            generated = []
            for idx, data in enumerate(iterator, 1):
                if iterator.done:
                    assert iterator.fetched == 3
                else:
                    assert iterator.fetched == (idx + prefetch_batches)
                generated.append(data)
            return generated

        assert generate() == expected
        # validate reset works properly.
        assert generate() == expected
        assert iterator.fetched == 3

    class EmptyIterDataset(IterableDataset):
        def __iter__(self):
            return iter([])

    dataloader = DataLoader(EmptyIterDataset())
    iterator = DataFetcher()
    iterator.setup(dataloader)
    assert list(iterator) == []


def test_misconfiguration_error():

    fetcher = DataFetcher()
    with pytest.raises(
        MisconfigurationException, match="The `dataloader_iter` isn't available outside the __iter__ context."
    ):
        loader = DataLoader(range(10))
        fetcher.setup(loader)
        assert fetcher.loaders[0] == loader
        fetcher.loader_iters

    iter(fetcher)
    assert fetcher.loader_iters


def get_cycles_per_ms() -> float:
    """
    Get 10 values and remove the 2 max and 2 min and return the avg.
    This is to avoid system disturbance that skew the results, e.g. the very first cuda call likely does a bunch of
    init, which takes much longer than subsequent calls.
    """

    def measure() -> float:
        """
        Measure and return approximate number of cycles per millisecond for `torch.cuda._sleep`
        Copied from: https://github.com/pytorch/pytorch/blob/v1.9.0/test/test_cuda.py#L81
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
EMB_SZ = 100
EMB_DIM = 64


class RandomIndicesDataset(Dataset):
    def __getitem__(self, index):
        return torch.randint(EMB_DIM, [BATCH_SIZE])

    def __len__(self):
        return 16


class RecommenderModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None
        self.local_embedding = torch.nn.Embedding(EMB_SZ, EMB_DIM)
        self.CYCLES_PER_MS = int(get_cycles_per_ms())

    def forward(self, indices: torch.Tensor):
        result = self.local_embedding(indices)
        return result

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # emulate heavy routine
        torch.cuda._sleep(self.CYCLES_PER_MS * 50)
        return batch

    def training_step_end(self, training_step_outputs):
        # emulate heavy routine
        torch.cuda._sleep(self.CYCLES_PER_MS * 50)
        return training_step_outputs

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomIndicesDataset(), batch_size=4)

    def val_dataloader(self):
        return DataLoader(RandomIndicesDataset(), batch_size=4)

    def test_dataloader(self):
        return DataLoader(RandomIndicesDataset(), batch_size=4)


@RunIf(min_gpus=1, min_torch="1.8.0")
def test_trainer_num_prefetch_batches(tmpdir):

    model = RecommenderModel()

    class CheckDataFetcher(Callback):
        def __init__(self, data_fetcher_cls: Type):
            self.data_fetcher_cls = data_fetcher_cls

        def on_batch_start(self, trainer, *_) -> None:
            assert isinstance(trainer.data_connector.train_data_fetcher, self.data_fetcher_cls)

    trainer_kwargs = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=1,
        limit_train_batches=4,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    with mock.patch.dict(os.environ, {"PL_INTER_BATCH_PARALLELISM": "1"}):
        t0 = time()
        trainer_kwargs["callbacks"] = CheckDataFetcher(InterBatchParallelDataFetcher)
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model)
        t1 = time()
        global_step = trainer.global_step

    torch.cuda.synchronize()

    t2 = time()
    trainer_kwargs["callbacks"] = CheckDataFetcher(DataFetcher)
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)
    t3 = time()

    assert global_step == trainer.global_step == 4
    ratio = (t3 - t2) / (t1 - t0)
    assert ratio > 1.1, ratio


@pytest.mark.parametrize("automatic_optimization", [False, True])
@RunIf(min_torch="1.8.0")
def test_fetching_dataloader_iter(automatic_optimization, tmpdir):
    class TestModel(BoringModel):
        def __init__(self, *args, automatic_optimization: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = automatic_optimization
            self.count = 0
            self.batches = []

        def training_step(self, dataloader_iter, batch_idx):
            assert self.count == batch_idx
            assert isinstance(self.trainer.data_connector.train_data_fetcher, DataLoaderIterDataFetcher)
            # fetch 2 batches
            self.batches.append(next(dataloader_iter))
            self.batches.append(next(dataloader_iter))

            batch = self.batches.pop(0)
            assert isinstance(batch, torch.Tensor) or batch is None
            self.count += 2
            if self.automatic_optimization:
                loss = super().training_step(batch, 0)
                with pytest.raises(MisconfigurationException, match="`batch_size` should be provided"):
                    self.log("train_loss", loss["loss"])
                self.log("train_loss", loss["loss"], batch_size=1)
            else:
                opt = self.optimizers()
                output = self(batch)
                loss = self.loss(batch, output)
                opt.zero_grad()
                loss.backward()
                opt.step()

        def training_epoch_end(self, *_):
            assert self.trainer.fit_loop.epoch_loop.batch_progress.current.ready == 33
            assert self.trainer.data_connector.train_data_fetcher.fetched == 64
            assert self.count == 64

    model = TestModel(automatic_optimization=automatic_optimization)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)

    # should be cleaned out !
    assert not hasattr(trainer.data_connector, "train_data_fetcher")


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

        pred = self.layer(batch_i)
        loss = self.loss(batch_i, pred)
        loss.backward()
        self.optimizers().step()
        self.optimizers().zero_grad()

        self.batch_i_handle = batch_ip1_handle
        self.num_batches_processed += 1

        return {"loss": loss, "is_last": is_last}

    def train_dataloader(self):
        return DataLoader(RandomDataset(BATCH_SIZE, DATASET_LEN))


def test_training_step_with_dataloader_access(tmpdir) -> None:
    """
    A baseline functional test for `training_step` with dataloader access.
    """
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = AsyncBoringModel()
    trainer.fit(m)
    assert m.num_batches_processed == DATASET_LEN, f"Expect all {DATASET_LEN} batches to be processed."


def test_stop_iteration(tmpdir) -> None:
    """
    Verify that when `StopIteration` is raised within `training_step`, `fit()`
    terminiates as expected.
    """
    EXPECT_NUM_BATCHES_PROCESSED = 2

    class TestModel(AsyncBoringModel):
        def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
            output = super().training_step(dataloader_iter)
            if self.num_batches_processed == EXPECT_NUM_BATCHES_PROCESSED:
                raise StopIteration
            return output

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = TestModel()
    trainer.fit(m)
    assert (
        m.num_batches_processed == EXPECT_NUM_BATCHES_PROCESSED
    ), "Expect {EXPECT_NUM_BATCHES_PROCESSED} batches to be processed."


def test_on_train_batch_start_overridden(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when
    `on_train_batch_start` is overridden on the `LightningModule`.
    """

    class InvalidModel(AsyncBoringModel):
        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            pass

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.raises(MisconfigurationException, match="The model hook `on_train_batch_start` is not compatible with"):
        trainer.fit(m)


def test_on_train_batch_end_overridden(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when
    `on_train_batch_end` is overridden on the `LightningModule`.
    """

    class InvalidModel(AsyncBoringModel):
        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            pass

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.raises(MisconfigurationException, match="The model hook `on_train_batch_end` is not compatible with"):
        trainer.fit(m)


def test_tbptt_split_batch_overridden(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when
    `tbptt_split_batch` is overridden on the `LightningModule`.
    """

    class InvalidModel(AsyncBoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.truncated_bptt_steps = 2

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.raises(MisconfigurationException, match="is incompatible with `truncated_bptt_steps > 0`."):
        trainer.fit(m)
