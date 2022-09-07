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
from unittest import mock

import pytest
import torch
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators.cuda import CUDAAccelerator, get_nvidia_gpu_stats


@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
def test_auto_device_count(_):
    assert CUDAAccelerator.auto_device_count() == 2


@RunIf(min_cuda_gpus=1)
def test_gpu_availability():
    assert CUDAAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be CUDA"):
        CUDAAccelerator().init_device(torch.device("cpu"))


@pytest.mark.parametrize(
    "devices,expected",
    [
        ([], []),
        ([1], [torch.device("cuda", 1)]),
        ([3, 1], [torch.device("cuda", 3), torch.device("cuda", 1)]),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert CUDAAccelerator.get_parallel_devices(devices) == expected


@RunIf(min_cuda_gpus=1)
def test_get_torch_gpu_stats(tmpdir):
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_stats = CUDAAccelerator().get_device_stats(current_device)
    fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@RunIf(min_cuda_gpus=1)
def test_get_nvidia_gpu_stats(tmpdir):
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_stats = get_nvidia_gpu_stats(current_device)
    fields = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@mock.patch("torch.cuda.set_device")
def test_set_cuda_device(set_device_mock):
    CUDAAccelerator().init_device(torch.device("cuda", 1))
    set_device_mock.assert_called_once_with(torch.device("cuda", 1))
