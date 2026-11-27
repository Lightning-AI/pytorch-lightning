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
import logging
import os
from unittest import mock

import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import AcceleratorRegistry, NPUAccelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf


@mock.patch("lightning.pytorch.accelerators.npu.num_npu_devices", return_value=2)
def test_auto_device_count(_):
    assert NPUAccelerator.auto_device_count() == 2


@RunIf(min_npu_gpus=1)
def test_npu_availability():
    assert NPUAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(MisconfigurationException, match="Device should be NPU"):
        NPUAccelerator().setup_device(torch.device("cpu"))


@RunIf(min_npu_gpus=1)
@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ([], []),
        ([1], [torch.device("npu", 1)]),
        ([3, 1], [torch.device("npu", 3), torch.device("npu", 1)]),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert NPUAccelerator.get_parallel_devices(devices) == expected


@mock.patch("lightning.pytorch.accelerators.npu.num_npu_devices", return_value=8)
def test_set_ascend_flags(_, caplog):
    with mock.patch.dict(os.environ, {}, clear=True):
        with caplog.at_level(logging.INFO, logger="lightning.pytorch.accelerators.npu"):
            NPUAccelerator.set_ascend_flags(0)
    assert "LOCAL_RANK: 0" in caplog.text
    assert "ASCEND_RT_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]" in caplog.text


@mock.patch("lightning.fabric.accelerators.npu.num_npu_devices", return_value=8)
def test_parse_devices(_):
    assert NPUAccelerator.parse_devices(-1) == list(range(8))
    assert NPUAccelerator.parse_devices(0) is None
    assert NPUAccelerator.parse_devices("2,3") == [2, 3]


def test_npu_accelerator_in_registry():
    assert "npu" in AcceleratorRegistry
    assert AcceleratorRegistry["npu"]["accelerator"] is NPUAccelerator


@RunIf(min_npu_gpus=1)
def test_trainer_npu_smoke():
    trainer = Trainer(accelerator="npu", devices=1)
    assert isinstance(trainer.accelerator, NPUAccelerator)
