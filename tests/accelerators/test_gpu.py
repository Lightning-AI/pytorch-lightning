from unittest import mock

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import GPUAccelerator
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(min_torch="1.8")
@RunIf(min_gpus=1)
def test_get_torch_gpu_stats(tmpdir):
    """Test GPU get_device_stats with Pytorch >= 1.8.0."""
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_stats = GPUAccelerator().get_device_stats(current_device)
    fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@RunIf(max_torch="1.7")
@RunIf(min_gpus=1)
def test_get_nvidia_gpu_stats(tmpdir):
    """Test GPU get_device_stats with Pytorch < 1.8.0."""
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_stats = GPUAccelerator().get_device_stats(current_device)
    fields = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@RunIf(gpus=1)
@mock.patch("torch.cuda.set_device")
def test_set_cuda_device(set_device_mock, tmpdir):
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
