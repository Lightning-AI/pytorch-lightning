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
from lightning.pytorch.accelerators import IPUAccelerator
from lightning.pytorch.accelerators.ipu import _IPU_AVAILABLE
from lightning.pytorch.strategies.ipu import IPUStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.trainer.connectors.test_accelerator_connector import mock_ipu_available

if _IPU_AVAILABLE:
    import poptorch


def test_auto_device_count():
    assert IPUAccelerator.auto_device_count() == 4


@pytest.mark.skipif(_IPU_AVAILABLE, reason="test requires non-IPU machine")
@mock.patch("lightning.pytorch.accelerators.ipu.IPUAccelerator.is_available", return_value=True)
def test_fail_if_no_ipus(_, tmpdir):
    with pytest.raises(MisconfigurationException, match="IPU Accelerator requires IPU devices to run"):
        Trainer(default_root_dir=tmpdir, accelerator="ipu", devices=1)


@RunIf(ipu=True)
def test_accelerator_selected(tmpdir):
    assert IPUAccelerator.is_available()
    trainer = Trainer(default_root_dir=tmpdir, accelerator="ipu", devices=1)
    assert isinstance(trainer.accelerator, IPUAccelerator)


def test_warning_if_ipus_not_used(monkeypatch, cuda_count_0):
    mock_ipu_available(monkeypatch)
    with pytest.warns(UserWarning, match="IPU available but not used"):
        Trainer(accelerator="cpu")


@RunIf(ipu=True)
def test_no_warning_strategy(tmpdir):
    with pytest.warns(None) as record:
        Trainer(default_root_dir=tmpdir, max_epochs=1, strategy=IPUStrategy(training_opts=poptorch.Options()))
    assert len(record) == 0


@RunIf(ipu=True)
def test_accelerator_ipu():
    trainer = Trainer(accelerator="ipu", devices=1)
    assert isinstance(trainer.accelerator, IPUAccelerator)

    trainer = Trainer(accelerator="ipu")
    assert isinstance(trainer.accelerator, IPUAccelerator)

    trainer = Trainer(accelerator="auto", devices=8)
    assert isinstance(trainer.accelerator, IPUAccelerator)


@RunIf(ipu=True)
def test_accelerator_ipu_with_devices():
    trainer = Trainer(accelerator="ipu", devices=8)
    assert isinstance(trainer.strategy, IPUStrategy)
    assert isinstance(trainer.accelerator, IPUAccelerator)
    assert trainer.num_devices == 8


@RunIf(ipu=True)
def test_accelerator_auto_with_devices_ipu():
    trainer = Trainer(accelerator="auto", devices=8)
    assert isinstance(trainer.accelerator, IPUAccelerator)
    assert trainer.num_devices == 8


@RunIf(ipu=True)
def test_strategy_choice_ipu_strategy():
    trainer = Trainer(strategy=IPUStrategy(), accelerator="ipu", devices=8)
    assert isinstance(trainer.strategy, IPUStrategy)


@RunIf(ipu=True)
def test_device_type_when_ipu_strategy_passed():
    trainer = Trainer(strategy=IPUStrategy(), accelerator="ipu", devices=8)
    assert isinstance(trainer.strategy, IPUStrategy)
    assert isinstance(trainer.accelerator, IPUAccelerator)


@RunIf(ipu=True)
def test_devices_auto_choice_ipu():
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.num_devices == 4
    assert isinstance(trainer.accelerator, IPUAccelerator)
