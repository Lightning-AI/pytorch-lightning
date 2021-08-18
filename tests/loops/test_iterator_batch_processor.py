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
from typing import Any, Iterator

import pytest
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tests.helpers import BoringModel, RandomDataset

_BATCH_SIZE = 32
_DATASET_LEN = 64


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
        return DataLoader(RandomDataset(_BATCH_SIZE, _DATASET_LEN))


def test_training_step_with_dataloader_access(tmpdir) -> None:
    """
    A baseline functional test for `training_step` with dataloader access.
    """
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = AsyncBoringModel()
    trainer.fit(m)
    assert m.num_batches_processed == _DATASET_LEN, f"Expect all {_DATASET_LEN} batches to be processed."


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
                raise StopIteration()
            return output

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = TestModel()
    trainer.fit(m)
    assert (
        m.num_batches_processed == EXPECT_NUM_BATCHES_PROCESSED
    ), "Expect {EXPECT_NUM_BATCHES_PROCESSED} batches to be processed."


def test_automatic_optimization_enabled(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when
    `automatic_optimization` is enabled.
    """
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = AsyncBoringModel()
    m.automatic_optimization = True
    with pytest.raises(MisconfigurationException):
        trainer.fit(m)


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
    with pytest.raises(MisconfigurationException):
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
    with pytest.raises(MisconfigurationException):
        trainer.fit(m)


def test_tbptt_split_batch_overridden(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when
    `tbptt_split_batch` is overridden on the `LightningModule`.
    """

    class InvalidModel(AsyncBoringModel):
        def tbptt_split_batch(self, batch, split_size):
            pass

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.raises(MisconfigurationException):
        trainer.fit(m)


def test_accumulate_grad_batches(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when
    `accumulate_grad_batches` is not set to 1.
    """
    trainer = Trainer(max_epochs=1, accumulate_grad_batches=2, default_root_dir=tmpdir)
    m = AsyncBoringModel()
    with pytest.raises(MisconfigurationException):
        trainer.fit(m)


def test_is_last_not_set(tmpdir) -> None:
    """
    Verify that a `MisconfigurationException` is raised when `training_step`
    doesn't include "is_last" in the result dict.
    """

    class InvalidModel(AsyncBoringModel):
        def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
            output = super().training_step(dataloader_iter)
            del output["is_last"]
            return output

    trainer = Trainer(max_epochs=1, accumulate_grad_batches=2, default_root_dir=tmpdir)
    m = InvalidModel()
    with pytest.raises(MisconfigurationException):
        trainer.fit(m)
