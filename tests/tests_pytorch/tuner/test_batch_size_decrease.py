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

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.tuner import Tuner


def _data():
    return torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24


class _SmallDatasetModel(LightningModule):
    def __init__(self, initial_batch_size, owner="model"):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)
        self.owner = owner
        if owner == "model":
            self.batch_size = initial_batch_size
        elif owner == "hparams":
            self.save_hyperparameters({"batch_size": initial_batch_size})
        self.seen = {stage: [] for stage in ("fit", "validate", "test", "predict")}

    def _current_size(self):
        if self.owner == "datamodule":
            return self.trainer.datamodule.batch_size
        if self.owner == "hparams":
            return self.hparams.batch_size
        return self.batch_size

    def _step(self, batch, stage):
        self.seen[stage].append((self._current_size(), len(batch)))
        return self.layer(batch).square().mean()

    def training_step(self, batch, batch_idx):
        return self._step(batch, "fit")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "validate")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        return self._step(batch, "predict")

    def train_dataloader(self):
        return DataLoader(_data(), batch_size=self._current_size())

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader
    predict_dataloader = train_dataloader

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _SmallDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(_data(), batch_size=self.batch_size)

    val_dataloader = train_dataloader


def _trainer(tmp_path, method="fit"):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if method == "validate" else 0,
    )


def _tune(trainer, model, mode, datamodule=None, method="fit", init_val=2):
    return Tuner(trainer).scale_batch_size(
        model,
        datamodule=datamodule,
        method=method,
        mode=mode,
        steps_per_trial=1,
        max_trials=2,
        init_val=init_val,
        margin=0.0,
    )


@pytest.mark.parametrize("mode", ["power", "binsearch"])
@pytest.mark.parametrize("owner", ["model", "hparams", "datamodule"])
@pytest.mark.parametrize("initial_batch_size", [8, 16])
def test_large_existing_batch_does_not_override_smaller_initial_value(tmp_path, mode, owner, initial_batch_size):
    model = _SmallDatasetModel(initial_batch_size, owner)
    datamodule = _SmallDataModule(initial_batch_size) if owner == "datamodule" else None
    trainer = _trainer(tmp_path)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    size = _tune(trainer, model, mode, datamodule)

    assert size == 4
    assert model._current_size() == 4
    assert model.seen["fit"] == [(2, 2), (4, 4)]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not list(tmp_path.glob(".scale_batch_size_*.ckpt"))

    reference = torch.nn.Linear(3, 1)
    reference.load_state_dict(model.layer.state_dict())
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference(_data()[:4]).square().mean().backward()
    optimizer.step()
    trainer.fit(model, datamodule=datamodule)
    for actual, expected in zip(model.layer.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["power", "binsearch"])
@pytest.mark.parametrize("method", ["fit", "validate", "test", "predict"])
def test_smaller_initial_value_is_used_for_each_tuner_entrypoint(tmp_path, mode, method):
    model = _SmallDatasetModel(16)
    trainer = _trainer(tmp_path, method)

    assert _tune(trainer, model, mode, method=method) == 4
    expected = [(2, 2)] * 4 + [(4, 4)] * 2 if method == "predict" else [(2, 2), (4, 4)]
    assert model.seen[method] == expected


@pytest.mark.parametrize("mode", ["power", "binsearch"])
@pytest.mark.parametrize("initial_batch_size", [1, 2])
def test_existing_smaller_batch_keeps_the_same_search(tmp_path, mode, initial_batch_size):
    model = _SmallDatasetModel(initial_batch_size)

    assert _tune(_trainer(tmp_path), model, mode) == 4
    assert model.seen["fit"] == [(2, 2), (4, 4)]


@pytest.mark.parametrize("mode", ["power", "binsearch"])
def test_a_second_search_can_start_below_the_previous_dataset_sized_batch(tmp_path, mode):
    model = _SmallDatasetModel(2)
    trainer = _trainer(tmp_path)
    assert _tune(trainer, model, mode, init_val=8) == 8
    model.seen["fit"].clear()

    assert _tune(trainer, model, mode, init_val=2) == 4
    assert model.seen["fit"] == [(2, 2), (4, 4)]
