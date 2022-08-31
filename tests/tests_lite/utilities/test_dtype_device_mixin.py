import pytest
import torch
from torch import nn as nn

from lightning_lite.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class SubSubModule(DeviceDtypeModuleMixin):
    pass


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = SubSubModule()


class TopModule(DeviceDtypeModuleMixin):
    def __init__(self) -> None:
        super().__init__()
        self.module = SubModule()


@pytest.mark.parametrize(
    "dst_device_str,dst_dtype",
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
def test_submodules_device_and_dtype(dst_device_str, dst_dtype):
    """Test that the device and dtype property updates propagate through mixed nesting of regular nn.Modules and
    the special modules of type DeviceDtypeModuleMixin (e.g. Metric or LightningModule)."""

    dst_device = torch.device(dst_device_str)

    model = TopModule()
    assert model.device == torch.device("cpu")
    model = model.to(device=dst_device, dtype=dst_dtype)
    # nn.Module does not have these attributes
    assert not hasattr(model.module, "_device")
    assert not hasattr(model.module, "_dtype")
    # device and dtype change should propagate down into all children
    assert model.device == model.module.module.device == dst_device
    assert model.dtype == model.module.module.dtype == dst_dtype


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


@RunIf(min_cuda_gpus=2)
def test_cuda_current_device():
    """Test that calling .cuda() moves the model to the correct device and respects current cuda device setting."""

    class CudaModule(DeviceDtypeModuleMixin):
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
