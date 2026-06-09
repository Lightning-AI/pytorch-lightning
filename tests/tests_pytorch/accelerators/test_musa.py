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

from unittest import mock

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import MUSAAccelerator
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


@RunIf(musa=True)
def test_musa_availability():
    assert MUSAAccelerator.is_available()


def test_warning_if_musa_not_used(musa_count_1):
    with pytest.warns(UserWarning, match="GPU available but not used"):
        Trainer(accelerator="cpu")


@RunIf(musa=True)
@pytest.mark.parametrize("accelerator_value", ["musa", MUSAAccelerator()])
def test_trainer_musa_accelerator(accelerator_value):
    trainer = Trainer(accelerator=accelerator_value, devices=1)
    assert isinstance(trainer.accelerator, MUSAAccelerator)
    assert trainer.num_devices == 1


@RunIf(musa=True)
@mock.patch("torch.musa.set_device")
def test_set_musa_device(set_device_mock, tmp_path, monkeypatch):
    monkeypatch.setenv("MUSA_DEVICE_ORDER", "PCI_BUS_ID")  # 或其他需要的值
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    set_device_mock.assert_called_once()
