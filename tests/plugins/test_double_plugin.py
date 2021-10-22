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
import pickle
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DoublePrecisionPlugin
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_7
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


class RandomFloatIntDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.float_data = torch.randn(length, size)
        self.int_data = torch.randint(10, (length, 1))

    def __getitem__(self, index):
        return self.float_data[index], self.int_data[index]

    def __len__(self):
        return self.len


class DoublePrecisionBoringModel(BoringModel):
    def training_step(self, batch, batch_idx):
        float_data, int_data = batch
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        assert float_data.dtype == torch.float64
        output = self(float_data)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        assert torch.tensor([0.0]).dtype == torch.float32
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        return self(batch)

    def on_fit_start(self):
        assert self.layer.weight.dtype == torch.float64

    def on_after_backward(self):
        assert self.layer.weight.grad.dtype == torch.float64

    def train_dataloader(self):
        dataset = RandomFloatIntDataset(32, 64)
        assert dataset.float_data.dtype == torch.float32  # Don't start with double data
        return DataLoader(dataset)

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class DoublePrecisionBoringModelNoForward(BoringModel):
    def training_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(batch, output)
        return {"y": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        return output

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class DoublePrecisionBoringModelComplexBuffer(BoringModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("complex_buffer", torch.tensor([1.2, 3.4j]), False)

    def on_fit_start(self):
        super().on_fit_start()
        # when the default floating point type is float64 the default complex type is complex128
        assert self.complex_buffer.dtype == torch.complex128
        # this hook is not wrapped. # TODO: should it be?
        assert torch.tensor([1.2, 3.4j]).dtype == torch.complex64

    def training_step(self, batch, batch_idx):
        assert torch.tensor([1.2, 3.4j]).dtype == torch.complex128
        return super().training_step(batch, batch_idx)


@pytest.mark.parametrize(
    "boring_model",
    [
        DoublePrecisionBoringModel,
        DoublePrecisionBoringModelNoForward,
        pytest.param(
            DoublePrecisionBoringModelComplexBuffer,
            marks=pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_7, reason="torch.complex not available"),
        ),
    ],
)
def test_double_precision(tmpdir, boring_model):
    trainer = Trainer(max_epochs=2, default_root_dir=tmpdir, fast_dev_run=2, precision=64, log_every_n_steps=1)
    with trainer.precision_plugin.autodtype():
        model = boring_model()
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)


@RunIf(min_gpus=2)
def test_double_precision_ddp(tmpdir):
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        strategy="ddp_spawn",
        gpus=2,
        fast_dev_run=2,
        precision=64,
        log_every_n_steps=1,
    )
    with trainer.precision_plugin.autodtype():
        model = DoublePrecisionBoringModel()
    trainer.fit(model)


def test_double_precision_pickle(tmpdir):
    model = BoringModel()
    plugin = DoublePrecisionPlugin()
    model, _, __ = plugin.connect(model, MagicMock(), MagicMock())
    pickle.dumps(model)


def test_double_precision_restores_dtype():
    class DummyException(BaseException):
        ...

    class Model(BoringModel):
        def training_step(self, batch, batch_idx):
            assert torch.get_default_dtype() == torch.double
            raise DummyException

    model = Model()
    trainer = Trainer(precision=64, num_sanity_val_steps=0)

    assert torch.get_default_dtype() == torch.float
    with pytest.raises(DummyException):
        trainer.fit(model)
    assert torch.get_default_dtype() == torch.float
