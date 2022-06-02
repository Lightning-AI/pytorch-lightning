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

import logging
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from pytorch_lightning.strategies.ipu import IPUStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomIterableDataset
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test


def test_num_stepping_batches_basic():
    """Test number of stepping batches in a general case."""
    max_epochs = 2
    trainer = Trainer(max_epochs=max_epochs)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == 64 * max_epochs


def test_num_stepping_batches_with_diff_multiple_grad_accum_factor():
    """Test that an error is raised if `Trainer` is configured with different gradient accumulation factors at
    different epochs."""
    grad_scheduler = GradientAccumulationScheduler(scheduling={7: 2})
    trainer = Trainer(callbacks=[grad_scheduler])
    with pytest.raises(MisconfigurationException, match="cannot be computed with different"):
        assert trainer.estimated_stepping_batches


def test_num_stepping_batches_raises_info_with_no_dataloaders_loaded(caplog):
    """Test that an info message is generated when dataloaders are loaded explicitly if they are not already
    configured."""
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    message = "to estimate number of stepping batches"
    trainer.reset_train_dataloader()
    with caplog.at_level(logging.INFO):
        assert trainer.estimated_stepping_batches == 64

    assert message not in caplog.text

    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    with caplog.at_level(logging.INFO):
        assert trainer.estimated_stepping_batches == 64

    assert message in caplog.text


def test_num_stepping_batches_iterable_dataset():
    """Test the stepping batches with iterable dataset configured with max steps."""
    max_steps = 1000
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    train_dl = DataLoader(RandomIterableDataset(size=7, count=1e10))
    trainer._data_connector.attach_data(model, train_dataloaders=train_dl)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == max_steps


def test_num_stepping_batches_infinite_training():
    """Test that stepping batches is "inf" when `Trainer` is configured for infinite training."""
    trainer = Trainer(max_steps=-1, max_epochs=-1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == float("inf")


def test_num_stepping_batches_with_max_steps():
    """Test stepping batches with `max_steps`."""
    max_steps = 7
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == max_steps


@pytest.mark.parametrize("accumulate_grad_batches,expected_steps", [(2, 32), (3, 22)])
def test_num_stepping_batches_accumulate_gradients(accumulate_grad_batches, expected_steps):
    """Test the total stepping batches when accumulation grad batches is configured."""
    trainer = Trainer(max_epochs=1, accumulate_grad_batches=accumulate_grad_batches)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == expected_steps


@pytest.mark.parametrize(
    ["trainer_kwargs", "estimated_steps"],
    [
        ({"strategy": "ddp", "num_nodes": 1}, 10),
        ({"strategy": "ddp", "num_nodes": 2}, 5),
        ({"strategy": "ddp", "num_nodes": 3}, 4),
        ({"strategy": "ddp", "num_nodes": 4}, 3),
        ({"strategy": "dp"}, 64),
        ({"strategy": "ddp2", "num_nodes": 1}, 64),
        ({"strategy": "ddp2", "num_nodes": 2}, 32),
        ({"strategy": "ddp2", "num_nodes": 3}, 22),
    ],
)
def test_num_stepping_batches_gpu(trainer_kwargs, estimated_steps, monkeypatch):
    """Test stepping batches with GPU strategies."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, devices=7, accelerator="gpu", **trainer_kwargs)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == estimated_steps


@RunIf(tpu=True)
@pl_multi_process_test
@pytest.mark.parametrize("devices,estimated_steps", [([1], 64), (8, 8)])
def test_num_stepping_batches_with_tpu(devices, estimated_steps):
    """Test stepping batches with TPU training which acts like DDP."""
    trainer = Trainer(accelerator="tpu", devices=devices, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == estimated_steps


@mock.patch("pytorch_lightning.accelerators.ipu.IPUAccelerator.is_available", return_value=True)
def test_num_stepping_batches_with_ipu(mock_ipu_acc_avail, monkeypatch):
    """Test stepping batches with IPU training which acts like DP."""
    import pytorch_lightning.strategies.ipu as ipu

    monkeypatch.setattr(ipu, "_IPU_AVAILABLE", True)
    trainer = Trainer(accelerator="ipu", devices=2, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert isinstance(trainer.strategy, IPUStrategy)
    assert trainer.estimated_stepping_batches == 64
