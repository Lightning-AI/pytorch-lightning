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
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import XPUAccelerator
from lightning.pytorch.accelerators.xpu import get_intel_gpu_stats
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


@RunIf(min_xpu_gpus=1)
def test_get_torch_gpu_stats():
    current_device = torch.device(f"xpu:{torch.xpu.current_device()}")
    gpu_stats = XPUAccelerator().get_device_stats(current_device)
    fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@RunIf(min_xpu_gpus=1)
def test_get_intel_gpu_stats():
    current_device = torch.device(f"xpu:{torch.xpu.current_device()}")
    gpu_stats = get_intel_gpu_stats(current_device)
    fields = [
        "GPU Utilization (%)",
        "GPU Memory Used (MiB)",
        "GPU Memory Utilization (%)",
        "GPU Core Temperature (°C)",
        "GPU Memory Temperature (°C)",
    ]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@RunIf(min_xpu_gpus=1)
@mock.patch("torch.xpu.set_device")
def test_set_xpu_device(set_device_mock, tmpdir):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    set_device_mock.assert_called_once()


@RunIf(min_xpu_gpus=1)
def test_gpu_availability():
    assert XPUAccelerator.is_available()


@RunIf(min_xpu_gpus=1)
def test_warning_if_gpus_not_used():
    with pytest.warns(UserWarning, match="GPU available but not used. Set `accelerator` and `devices`"):
        Trainer(accelerator="cpu")
