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
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomIterableDataset
from tests.helpers.runif import RunIf
from tests.helpers.utils import no_warning_call, pl_multi_process_test


def test_num_optimization_steps_basic():
    """Test number of optimization steps in a general case."""
    max_epochs = 2
    trainer = Trainer(max_epochs=max_epochs)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == 64 * max_epochs


def test_num_optimization_steps_with_diff_multiple_grad_accum_factor():
    """Test that an error is raised if `Trainer` is configured with different gradient accumulation factors at
    different epochs."""
    grad_scheduler = GradientAccumulationScheduler(scheduling={7: 2})
    trainer = Trainer(callbacks=[grad_scheduler])
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    with pytest.raises(MisconfigurationException, match="cannot be computed with different"):
        assert trainer.estimated_num_optimization_steps


def test_num_optimization_steps_raises_warning_with_no_dataloaders_loaded():
    """Test that a warning is raised when dataloaders are loaded explicitly if they are not already configured."""
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    trainer.reset_train_dataloader()
    with no_warning_call(UserWarning, match="to estimate number of optimization steps"):
        assert trainer.estimated_num_optimization_steps == 64

    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    with pytest.warns(UserWarning, match="to estimate number of optimization steps"):
        assert trainer.estimated_num_optimization_steps == 64


def test_num_optimization_steps_iterable_dataset():
    """Test the optimization steps with iterable dataset configured with max steps."""
    max_steps = 1000
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    train_dl = DataLoader(RandomIterableDataset(size=7, count=1e10))
    trainer._data_connector.attach_data(model, train_dataloaders=train_dl)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == max_steps


def test_num_optimization_steps_infinite_training():
    """Test that optimization steps is "inf" when `Trainer` is configured for infinite training."""
    trainer = Trainer(max_steps=-1, max_epochs=-1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == float("inf")


def test_num_optimization_steps_with_max_steps():
    """Test optimization steps with `max_steps`."""
    max_steps = 7
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == max_steps


@pytest.mark.parametrize("accumulate_grad_batches,expected_steps", [(2, 32), (3, 22)])
def test_num_optimization_steps_accumulate_gradients(accumulate_grad_batches, expected_steps):
    """Test the total optimization steps when accumulation grad batches is configured."""
    trainer = Trainer(max_epochs=1, accumulate_grad_batches=accumulate_grad_batches)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == expected_steps


@pytest.mark.parametrize("num_nodes,estimated_steps", [(1, 10), (2, 5), (3, 4), (4, 3)])
def test_num_optimization_steps_ddp(num_nodes, estimated_steps, monkeypatch):
    """Test optimization steps with DDP."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, num_nodes=num_nodes, devices=7, accelerator="gpu", strategy="ddp")
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == estimated_steps


@pytest.mark.parametrize("num_nodes,estimated_steps", [(1, 64), (2, 32), (3, 22)])
def test_num_optimization_steps_ddp2(num_nodes, estimated_steps, monkeypatch):
    """Test optimization steps with DDP2."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, num_nodes=num_nodes, devices=7, accelerator="gpu", strategy="ddp2")
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == estimated_steps


def test_num_optimization_steps_dp(monkeypatch):
    """Test optimization steps with DP."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    trainer = Trainer(max_epochs=1, devices=7, accelerator="gpu", strategy="dp")
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == 64


@RunIf(tpu=True)
@pl_multi_process_test
@pytest.mark.parametrize("devices,estimated_steps", [([1], 64), (8, 8)])
def test_num_optimization_steps_with_tpu(devices, estimated_steps):
    """Test optimization steps with TPU training which acts like DDP."""
    trainer = Trainer(accelerator="tpu", devices=devices, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == estimated_steps


def test_num_optimization_steps_with_ipu(monkeypatch):
    """Test optimization steps with IPU training which acts like DP."""
    import pytorch_lightning.strategies.ipu as ipu
    from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

    monkeypatch.setattr(ipu, "_IPU_AVAILABLE", True)
    monkeypatch.setattr(AcceleratorConnector, "has_ipu", True)
    trainer = Trainer(accelerator="ipu", devices=2, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_num_optimization_steps == 64
