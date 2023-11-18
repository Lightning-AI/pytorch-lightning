import pytest
import torch
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from torch import nn as nn

from tests_fabric.helpers.runif import RunIf


class SubSubModule(_DeviceDtypeModuleMixin):
    pass


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = SubSubModule()


class TopModule(_DeviceDtypeModuleMixin):
    def __init__(self) -> None:
        super().__init__()
        self.module = SubModule()


@pytest.mark.parametrize(
    ("dst_device_str", "dst_type"),
    [
        ("cpu", torch.half),
        ("cpu", torch.float),
        ("cpu", torch.double),
        pytest.param("cuda:0", torch.half, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda:0", torch.float, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda:0", torch.double, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps:0", torch.float, marks=RunIf(mps=True)),  # double and half are not yet supported.
    ],
)
@RunIf(min_cuda_gpus=1)
def test_submodules_device_and_dtype(dst_device_str, dst_type):
    """Test that the device and dtype property updates propagate through mixed nesting of regular nn.Modules and the
    special modules of type DeviceDtypeModuleMixin (e.g. Metric or LightningModule)."""
    dst_device = torch.device(dst_device_str)
    model = TopModule()
    assert model.device == torch.device("cpu")
    model = model.to(device=dst_device, dtype=dst_type)
    # nn.Module does not have these attributes
    assert not hasattr(model.module, "_device")
    assert not hasattr(model.module, "_dtype")
    # device and dtype change should propagate down into all children
    assert model.device == model.module.module.device == dst_device
    assert model.dtype == model.module.module.dtype == dst_type


@pytest.mark.parametrize(
    "device",
    [
        None,  # explicitly call without an index to see if the returning device contains an index
        0,
        torch.device("cuda", 0),
    ],
)
@RunIf(min_cuda_gpus=1)
def test_cuda_device(device):
    model = TopModule()

    model.cuda(device)

    device = model.device
    assert device.type == "cuda"
    assert device.index is not None
    assert device.index == torch.cuda.current_device()


@RunIf(min_cuda_gpus=1)
def test_cpu_device():
    model = SubSubModule().cuda()
    assert model.device.type == "cuda"
    assert model.device.index == 0
    model.cpu()
    assert model.device.type == "cpu"
    assert model.device.index is None


@RunIf(min_cuda_gpus=2)
def test_cuda_current_device():
    """Test that calling .cuda() moves the model to the correct device and respects current cuda device setting."""

    class CudaModule(_DeviceDtypeModuleMixin):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(1, 1)

    model = CudaModule()

    torch.cuda.set_device(0)
    model.cuda(1)
    assert model.device == torch.device("cuda", 1)
    assert model.layer.weight.device == torch.device("cuda", 1)

    torch.cuda.set_device(1)
    model.cuda()  # model is already on device 1, and calling .cuda() without device index should not move model
    assert model.device == torch.device("cuda", 1)
    assert model.layer.weight.device == torch.device("cuda", 1)


class ExampleModule(_DeviceDtypeModuleMixin):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)


def test_to_combinations():
    module = ExampleModule(torch.rand(3, 4))
    # sanity check
    assert module.weight.shape == (3, 4)
    assert module.weight.dtype is torch.float32
    # positional dtype
    module.to(torch.double)
    assert module.weight.dtype is torch.float64
    # positional device
    module.to("cpu", dtype=torch.half, non_blocking=True)
    assert module.weight.dtype is torch.float16
    assert module.device == torch.device("cpu")
    assert module.dtype is torch.float16


def test_dtype_conversions():
    module = ExampleModule(torch.tensor(1))
    # different dtypes
    assert module.weight.dtype is torch.int64
    assert module.dtype is torch.float32
    # `.double()` skips non floating points
    module.double()
    assert module.weight.dtype is torch.int64
    assert module.dtype is torch.float64
    # but `type` doesn't
    module.type(torch.float)
    assert module.weight.dtype is torch.float32
    assert module.dtype is torch.float32
    # now, test the rest
    module.float()
    assert module.weight.dtype is torch.float32
    assert module.dtype is torch.float32
    module.half()
    assert module.weight.dtype is torch.float16
    assert module.dtype is torch.float16
