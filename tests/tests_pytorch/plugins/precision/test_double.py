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
import pickle
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.plugins.precision.double import DoublePrecision
from torch.utils.data import DataLoader, Dataset

from tests_pytorch.helpers.runif import RunIf


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
        float_data, _ = batch
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        assert float_data.dtype == torch.float64
        return super().training_step(float_data, batch_idx)

    def on_train_epoch_end(self):
        assert torch.tensor([0.0]).dtype == torch.float32

    def validation_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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
        loss = self.loss(output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(output)
        return {"y": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        return output

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class DoublePrecisionBoringModelComplexBuffer(BoringModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("complex_buffer_wrong", torch.complex(torch.rand(10), torch.rand(10)), persistent=False)

    def configure_model(self) -> None:
        self.register_buffer("complex_buffer_right", torch.complex(torch.rand(10), torch.rand(10)), persistent=False)

    def on_fit_start(self):
        # when the default floating point type is float64 the default complex type is complex128, as long as it is
        # initialized under the precision context manager, because `model.to(double)` will not convert properly
        assert self.complex_buffer_wrong.dtype == torch.complex64
        assert self.complex_buffer_right.dtype == torch.complex128
        # this hook is not wrapped
        assert torch.tensor([1.2, 3.4j]).dtype == torch.complex64

    def training_step(self, batch, batch_idx):
        assert torch.tensor([1.2, 3.4j]).dtype == torch.complex128
        return super().training_step(batch, batch_idx)


@RunIf(mps=False)  # mps does not support float64
@pytest.mark.parametrize(
    "boring_model",
    [
        DoublePrecisionBoringModel,
        DoublePrecisionBoringModelNoForward,
        DoublePrecisionBoringModelComplexBuffer,
    ],
)
def test_double_precision(tmp_path, boring_model):
    model = boring_model()

    trainer = Trainer(max_epochs=2, default_root_dir=tmp_path, fast_dev_run=2, precision="64-true", log_every_n_steps=1)
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)


@RunIf(min_cuda_gpus=2)
def test_double_precision_ddp(tmp_path):
    model = DoublePrecisionBoringModel()

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=2,
        fast_dev_run=2,
        precision="64-true",
        log_every_n_steps=1,
    )
    trainer.fit(model)
    trainer.validate(model)


def test_double_precision_pickle():
    model = BoringModel()
    plugin = DoublePrecision()
    model, _, __ = plugin.connect(model, MagicMock(), MagicMock())
    pickle.dumps(model)


def test_convert_module():
    plugin = DoublePrecision()
    model = BoringModel()
    assert model.layer.weight.dtype == model.layer.bias.dtype == torch.float32
    model = plugin.convert_module(model)
    assert model.layer.weight.dtype == model.layer.bias.dtype == torch.float64


def test_module_init_context():
    plugin = DoublePrecision()
    with plugin.module_init_context():
        model = torch.nn.Linear(2, 2)
        assert torch.get_default_dtype() == torch.double
    assert model.weight.dtype == torch.double
