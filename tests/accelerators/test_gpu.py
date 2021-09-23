import torch

from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from tests.helpers.runif import RunIf


@RunIf(min_torch="1.8")
@RunIf(min_gpus=1)
def test_get_torch_gpu_stats(tmpdir):
    """Test GPU get_device_stats with Pytorch >= 1.8.0."""
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    GPUAccel = GPUAccelerator(
        training_type_plugin=DataParallelPlugin(parallel_devices=[current_device]), precision_plugin=PrecisionPlugin()
    )
    gpu_stats = GPUAccel.get_device_stats(current_device)
    fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())


@RunIf(max_torch="1.7")
@RunIf(min_gpus=1)
def test_get_nvidia_gpu_stats(tmpdir):
    """Test GPU get_device_stats with Pytorch < 1.8.0."""
    current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    GPUAccel = GPUAccelerator(
        training_type_plugin=DataParallelPlugin(parallel_devices=[current_device]), precision_plugin=PrecisionPlugin()
    )
    gpu_stats = GPUAccel.get_device_stats(current_device)
    fields = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())
