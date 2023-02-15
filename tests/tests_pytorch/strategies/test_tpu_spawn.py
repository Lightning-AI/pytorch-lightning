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
import os
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import SingleDeviceStrategy, TPUSpawnStrategy
from tests_pytorch.helpers.dataloaders import CustomNotImplementedErrorDataloader
from tests_pytorch.helpers.runif import RunIf


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
    (
        ("train_dataloaders", _loader_no_len),
        ("val_dataloaders", _loader_no_len),
        ("test_dataloaders", _loader_no_len),
        ("predict_dataloaders", _loader_no_len),
        ("val_dataloaders", [_loader, _loader_no_len]),
    ),
)
def test_error_iterable_dataloaders_passed_to_fit(keyword, value, monkeypatch):
    trainer = Trainer()
    model = BoringModelNoDataloaders()
    strategy = SingleDeviceStrategy(accelerator=Mock())
    trainer._accelerator_connector.strategy = strategy
    process_dataloader_mock = Mock()
    monkeypatch.setattr(strategy, "process_dataloader", process_dataloader_mock)

    if "train" in keyword:
        fn = trainer.reset_train_dataloader
    elif "val" in keyword:
        fn = trainer.reset_val_dataloader
    elif "test" in keyword:
        fn = trainer.reset_test_dataloader
    else:
        fn = trainer.reset_predict_dataloader

    trainer._data_connector.attach_dataloaders(model, **{keyword: value})
    fn(model)

    expected = len(value) if isinstance(value, list) else 1
    assert process_dataloader_mock.call_count == expected


def test_error_process_iterable_dataloader(xla_available):
    strategy = TPUSpawnStrategy(MagicMock())
    with pytest.raises(TypeError, match="TPUs do not currently support"):
        strategy.process_dataloader(_loader_no_len)


class BoringModelTPU(BoringModel):
    def on_train_start(self) -> None:
        # assert strategy attributes for device setting
        assert self.device == torch.device("xla", index=1)
        assert os.environ.get("PT_XLA_DEBUG") == "1"


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_one_core():
    """Tests if device/debug flag is set correctly when training and after teardown for TPUSpawnStrategy."""
    model = BoringModelTPU()
    trainer = Trainer(accelerator="tpu", devices=1, fast_dev_run=True, strategy=TPUSpawnStrategy(debug=True))
    assert isinstance(trainer.strategy, TPUSpawnStrategy)
    trainer.fit(model)
    assert "PT_XLA_DEBUG" not in os.environ
