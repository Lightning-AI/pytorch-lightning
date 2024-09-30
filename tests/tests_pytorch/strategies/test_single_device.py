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
from unittest.mock import MagicMock, Mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import SingleDeviceStrategy
from torch.utils.data import DataLoader

from tests_pytorch.helpers.dataloaders import CustomNotImplementedErrorDataloader
from tests_pytorch.helpers.runif import RunIf


def test_single_cpu():
    """Tests if device is set correctly for single CPU strategy."""
    trainer = Trainer(accelerator="cpu")
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device("cuda:0")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(min_cuda_gpus=1, skip_windows=True)
def test_single_gpu():
    """Tests if device is set correctly when training and after teardown for single GPU strategy.

    Cannot run this test on MPS due to shared memory not allowing dedicated measurements of GPU memory utilization.

    """
    trainer = Trainer(accelerator="gpu", devices=1, fast_dev_run=True)
    # assert training strategy attributes for device setting
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("cuda:0")

    model = BoringModelGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory


class MockOptimizer: ...


def test_strategy_pickle():
    strategy = SingleDeviceStrategy("cpu")
    optimizer = MockOptimizer()

    strategy.optimizers = [optimizer]
    assert isinstance(strategy.optimizers[0], MockOptimizer)
    assert isinstance(strategy._lightning_optimizers[0], LightningOptimizer)

    state = pickle.dumps(strategy)
    # dumping did not get rid of the lightning optimizers
    assert isinstance(strategy._lightning_optimizers[0], LightningOptimizer)
    strategy_reloaded = pickle.loads(state)
    # loading restores the lightning optimizers
    assert isinstance(strategy_reloaded._lightning_optimizers[0], LightningOptimizer)


class BoringModelNoDataloaders(BoringModel):
    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError


_loader = DataLoader(RandomDataset(32, 64))
_loader_no_len = CustomNotImplementedErrorDataloader(_loader)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("train_dataloaders", _loader_no_len),
        ("val_dataloaders", _loader_no_len),
        ("test_dataloaders", _loader_no_len),
        ("predict_dataloaders", _loader_no_len),
        ("val_dataloaders", [_loader, _loader_no_len]),
    ],
)
def test_process_dataloader_gets_called_as_expected(keyword, value, monkeypatch):
    trainer = Trainer()
    model = BoringModelNoDataloaders()
    strategy = SingleDeviceStrategy(accelerator=Mock())
    strategy.connect(model)
    trainer._accelerator_connector.strategy = strategy
    process_dataloader_mock = MagicMock()
    monkeypatch.setattr(strategy, "process_dataloader", process_dataloader_mock)

    if "train" in keyword:
        fn = trainer.fit_loop.setup_data
    elif "val" in keyword:
        fn = trainer.validate_loop.setup_data
    elif "test" in keyword:
        fn = trainer.test_loop.setup_data
    else:
        fn = trainer.predict_loop.setup_data

    trainer._data_connector.attach_dataloaders(model, **{keyword: value})
    fn()

    expected = len(value) if isinstance(value, list) else 1
    assert process_dataloader_mock.call_count == expected
