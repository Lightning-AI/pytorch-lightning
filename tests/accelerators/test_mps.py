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

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import MPSAccelerator
from tests.helpers.runif import RunIf


@RunIf(mps=True)
def test_get_mps_stats():
    current_device = torch.device("mps")
    device_stats = MPSAccelerator().get_device_stats(current_device)
    fields = ["M1_vm_percent", "M1_percent", "M1_swap_percent"]

    for f in fields:
        assert any(f in h for h in device_stats.keys())


@RunIf(mps=True)
def test_mps_availability():
    assert MPSAccelerator.is_available()


@RunIf(mps=True)
def test_warning_if_mps_not_used():
    with pytest.warns(UserWarning, match="MPS available but not used. Set `accelerator` and `devices`"):
        Trainer()


@RunIf(mps=True)
@pytest.mark.parametrize("accelerator_value", ["mps", MPSAccelerator()])
def test_trainer_mps_accelerator(accelerator_value):
    trainer = Trainer(accelerator=accelerator_value)
    assert isinstance(trainer.accelerator, MPSAccelerator)
    assert trainer.num_devices == 1
    assert trainer.strategy.root_device.type == "mps"
