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
from unittest.mock import MagicMock

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import DeviceSummary
from lightning.pytorch.demos.boring_classes import BoringModel


def test_device_summary_callback_present_trainer():
    trainer = Trainer()
    assert any(isinstance(cb, DeviceSummary) for cb in trainer.callbacks)

    trainer = Trainer(callbacks=DeviceSummary())
    assert any(isinstance(cb, DeviceSummary) for cb in trainer.callbacks)


def test_device_summary_callback_with_enable_device_summary_false():
    trainer = Trainer(enable_device_summary=False)
    assert not any(isinstance(cb, DeviceSummary) for cb in trainer.callbacks)


def test_device_summary_callback_with_enable_device_summary_true():
    trainer = Trainer(enable_device_summary=True)
    assert any(isinstance(cb, DeviceSummary) for cb in trainer.callbacks)


def test_device_summary_callback_with_barebones():
    trainer = Trainer(barebones=True)
    assert not any(isinstance(cb, DeviceSummary) for cb in trainer.callbacks)


def test_device_summary_callback_with_barebones_error():
    with pytest.raises(ValueError, match="Device summary can impact raw speed"):
        Trainer(barebones=True, enable_device_summary=True)


def test_device_summary_callback_logging(tmp_path):
    model = BoringModel()
    trainer = MagicMock()
    callback = DeviceSummary(show_warnings=True)

    # Mock accelerator
    trainer.accelerator = MagicMock()
    trainer.num_devices = 1

    # Call setup
    callback.setup(trainer, model, "fit")
    assert callback._logged is True

    # Call again to verify it is only logged once
    callback.setup(trainer, model, "fit")
