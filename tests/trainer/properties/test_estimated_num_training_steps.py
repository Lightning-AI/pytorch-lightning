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

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel, RandomIterableDataset
from tests.helpers.runif import RunIf
from tests.helpers.utils import no_warning_call, pl_multi_process_test


def test_num_training_steps_basic():
    max_epochs = 2
    trainer = Trainer(max_epochs=max_epochs)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == 64 * max_epochs


def test_num_training_steps_raises_warning_with_no_dataloaders_loaded():
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    trainer.reset_train_dataloader()
    with no_warning_call(UserWarning, match="to estimate number of training steps"):
        assert trainer.estimated_num_training_steps == 64

    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    with pytest.warns(UserWarning, match="to estimate number of training steps"):
        assert trainer.estimated_num_training_steps == 64


def test_num_training_steps_iterable_dataset():
    max_steps = 1000
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    train_dl = DataLoader(RandomIterableDataset(size=7, count=1e10))
    trainer._data_connector.attach_data(model, train_dataloaders=train_dl)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == max_steps


def test_num_training_steps_infinite_training():
    trainer = Trainer(max_steps=-1, max_epochs=-1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == float("inf")


def test_num_training_steps_with_max_steps():
    max_steps = 7
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == max_steps


@pytest.mark.parametrize("accumulate_grad_batches,expected_steps", [(2, 32), (3, 22)])
def test_num_training_steps_accumulate_gradients(accumulate_grad_batches, expected_steps):
    trainer = Trainer(max_epochs=1, accumulate_grad_batches=accumulate_grad_batches)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == expected_steps


@pytest.mark.parametrize("num_nodes,estimated_steps", [(1, 10), (2, 5), (3, 4), (4, 3)])
def test_num_training_steps_ddp(num_nodes, estimated_steps, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, num_nodes=num_nodes, devices=7, accelerator="gpu", strategy="ddp")
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == estimated_steps


@pytest.mark.parametrize("num_nodes,estimated_steps", [(1, 64), (2, 32), (3, 22)])
def test_num_training_steps_ddp2(num_nodes, estimated_steps, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, num_nodes=num_nodes, devices=7, accelerator="gpu", strategy="ddp2")
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == estimated_steps


def test_num_training_steps_dp(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, devices=7, accelerator="gpu", strategy="dp")
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == 64


@RunIf(tpu=True)
@pl_multi_process_test
@pytest.mark.parametrize("devices,estimated_steps", [([1], 64), (8, 8)])
def test_num_training_steps_with_tpu(devices, estimated_steps):
    trainer = Trainer(accelerator="tpu", devices=devices, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == estimated_steps


def test_num_training_steps_with_ipu(monkeypatch):
    import pytorch_lightning.strategies.ipu as ipu
    from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

    monkeypatch.setattr(ipu, "_IPU_AVAILABLE", True)
    monkeypatch.setattr(AcceleratorConnector, "has_ipu", True)
    trainer = Trainer(accelerator="ipu", devices=2, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_training_steps == 64
