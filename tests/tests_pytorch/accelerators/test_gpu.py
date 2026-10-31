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
import multiprocessing
from unittest import mock

import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.accelerators.cuda import get_nvidia_gpu_stats
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


@RunIf(min_cuda_gpus=1)
def test_get_torch_gpu_stats():
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_stats = CUDAAccelerator().get_device_stats(current_device)
    fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]

    for f in fields:
        assert any(f in h for h in gpu_stats)


@RunIf(min_cuda_gpus=1)
def test_get_nvidia_gpu_stats():
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_stats = get_nvidia_gpu_stats(current_device)
    fields = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]

    for f in fields:
        assert any(f in h for h in gpu_stats)


@RunIf(min_cuda_gpus=1)
@mock.patch("torch.cuda.set_device")
def test_set_cuda_device(set_device_mock, tmp_path):
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


@RunIf(min_cuda_gpus=1)
def test_gpu_availability():
    assert CUDAAccelerator.is_available()


def test_warning_if_gpus_not_used(cuda_count_1):
    with pytest.warns(UserWarning, match="GPU available but not used"):
        Trainer(accelerator="cpu")


def _assert_set_device_precedes_lazy_init():
    """Assert `setup_device` selects the device before anything initializes CUDA.

    Only meaningful in a process where CUDA has not been initialized yet.

    """
    mock_set_device = mock.MagicMock(wraps=torch.cuda.set_device)
    mock_lazy_init = mock.MagicMock(wraps=torch.cuda._lazy_init)

    mock_manager = mock.MagicMock()
    mock_manager.attach_mock(mock_set_device, "set_device")
    mock_manager.attach_mock(mock_lazy_init, "_lazy_init")

    device = torch.device("cuda:0")

    with (
        mock.patch("torch.cuda.set_device", new=mock_set_device),
        mock.patch("torch.cuda._lazy_init", new=mock_lazy_init),
    ):
        CUDAAccelerator().setup_device(device)

    assert mock_manager.mock_calls[0] == mock.call.set_device(device)
    assert mock_manager.mock_calls[1] == mock.call._lazy_init()


@RunIf(min_cuda_gpus=1)
def test_setup_device_calls_set_device_before_lazy_init():
    # spawn a fresh process so the check is not invalidated by CUDA already being initialized
    spawn_context = multiprocessing.get_context("spawn")
    with spawn_context.Pool(processes=1) as pool:
        pool.apply(_assert_set_device_precedes_lazy_init)


@mock.patch("lightning.pytorch.accelerators.cuda._check_cuda_matmul_precision")
@mock.patch("torch.cuda.set_device")
def test_setup_device_sets_device_before_matmul_precision_check(set_device_mock, matmul_check_mock):
    """The matmul precision check may initialize CUDA, so the device must be selected first."""
    manager = mock.MagicMock()
    manager.attach_mock(set_device_mock, "set_device")
    manager.attach_mock(matmul_check_mock, "check_matmul_precision")

    device = torch.device("cuda", 3)
    CUDAAccelerator().setup_device(device)

    assert manager.mock_calls == [mock.call.set_device(device), mock.call.check_matmul_precision(device)]
