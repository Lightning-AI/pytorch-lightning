import pytest
import torch
import torch.nn as nn

from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from tests.base import EvalModelTemplate


class SubSubModule(DeviceDtypeModuleMixin, nn.Module):
    pass


class SubModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.sub_sub_module = SubSubModule()


class TopModule(EvalModelTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_module = SubModule()


@pytest.mark.parametrize(['dst_dtype'], [
    pytest.param(torch.float),
    pytest.param(torch.double),
    pytest.param(torch.half),
])
@pytest.mark.parametrize(['dst_device'], [
    pytest.param(torch.device('cpu')),
    pytest.param(torch.device('cuda')),
    pytest.param(torch.device('cuda', 0)),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_submodules_device_and_dtype(dst_device, dst_dtype):
    """
    Test that the device and dtype property updates propagate through mixed nesting of regular
    nn.Modules and the special modules of type DeviceDtypeModuleMixin (e.g. Metric or LightningModule).
    """

    model = TopModule()
    assert model.device == torch.device('cpu')
    model = model.to(device=dst_device, dtype=dst_dtype)
    assert model.device == model.sub_module.sub_sub_module.device == dst_device
    assert model.dtype == model.sub_module.sub_sub_module.dtype == dst_dtype
