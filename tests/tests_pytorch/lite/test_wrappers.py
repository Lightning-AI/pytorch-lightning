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
from unittest.mock import ANY, Mock

import pytest
import torch
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteDataLoader, _LiteModule, _LiteOptimizer
from tests_pytorch.helpers.runif import RunIf


class EmptyLite(LightningLite):
    def run(self):
        pass


def test_lite_module_wraps():
    """Test that the wrapped module is accessible via the property."""
    module = Mock()
    assert _LiteModule(module, Mock()).module is module

    wrapped_module = Mock()
    original_module = Mock()
    assert _LiteModule(wrapped_module, Mock(), original_module=original_module).module is original_module


def test_lite_module_attribute_lookup():
    """Test that attribute lookup passes through to the original model when possible."""

    class OriginalModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 3)
            self.attribute = 1

        def method(self):
            return 2

    original_module = OriginalModule()

    class ModuleWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapped = original_module

    wrapped_module = ModuleWrapper()

    lite_module = _LiteModule(wrapped_module, Mock(), original_module=original_module)
    assert lite_module.attribute == 1
    assert lite_module.layer is original_module.layer
    assert lite_module.method() == 2
    assert lite_module.forward.__self__.__class__ == _LiteModule

    with pytest.raises(AttributeError):
        _ = lite_module.not_exists


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "precision, input_type, expected_type",
    [
        (32, torch.float16, torch.float32),
        (32, torch.float32, torch.float32),
        (32, torch.float64, torch.float32),
        (32, torch.int, torch.int),
        (16, torch.float32, torch.float16),
        (16, torch.float64, torch.float16),
        (16, torch.long, torch.long),
        pytest.param("bf16", torch.float32, torch.bfloat16, marks=RunIf(min_torch="1.10", bf16_cuda=True)),
        pytest.param("bf16", torch.float64, torch.bfloat16, marks=RunIf(min_torch="1.10", bf16_cuda=True)),
        pytest.param("bf16", torch.bool, torch.bool, marks=RunIf(min_torch="1.10", bf16_cuda=True)),
    ],
)
def test_lite_module_forward_conversion(precision, input_type, expected_type):
    """Test that the LiteModule performs autocasting on the input tensors and during forward()."""
    lite = EmptyLite(precision=precision, accelerator="gpu", devices=1)
    device = torch.device("cuda", 0)

    def check_autocast(forward_input):
        assert precision != 16 or torch.is_autocast_enabled()
        return forward_input

    module = Mock(wraps=torch.nn.Identity(), side_effect=check_autocast)
    lite_module = _LiteModule(module, lite._precision_plugin).to(device)
    out = lite_module(torch.tensor([1, 2, 3], dtype=input_type, device=device))
    assert module.call_args[0][0].dtype == expected_type
    assert out.dtype == input_type or out.dtype == torch.get_default_dtype()


@pytest.mark.parametrize(
    "device", [torch.device("cpu"), pytest.param(torch.device("cuda", 0), marks=RunIf(min_cuda_gpus=1))]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lite_module_device_dtype_propagation(device, dtype):
    """Test that the LiteModule propagates device and dtype properties to its submodules (e.g. torchmetrics)."""

    class DeviceModule(DeviceDtypeModuleMixin):
        pass

    device_module = DeviceModule()
    lite_module = _LiteModule(device_module, Mock())
    lite_module.to(device)
    assert device_module.device == device
    assert lite_module.device == device

    lite_module.to(dtype)
    assert device_module.dtype == dtype
    assert lite_module.dtype == dtype


def test_lite_dataloader_iterator():
    """Test that the iteration over a LiteDataLoader wraps the iterator of the underlying dataloader (no automatic
    device placement)."""
    dataloader = DataLoader(range(5), batch_size=2)
    lite_dataloader = _LiteDataLoader(dataloader)
    assert len(lite_dataloader) == len(dataloader) == 3

    iterator = iter(dataloader)
    lite_iterator = iter(lite_dataloader)

    assert torch.equal(next(iterator), next(lite_iterator))
    assert torch.equal(next(iterator), next(lite_iterator))
    assert torch.equal(next(iterator), next(lite_iterator))

    with pytest.raises(StopIteration):
        next(iterator)

    with pytest.raises(StopIteration):
        next(lite_iterator)


@pytest.mark.parametrize(
    "src_device, dest_device",
    [
        (torch.device("cpu"), torch.device("cpu")),
        pytest.param(torch.device("cpu"), torch.device("cuda", 0), marks=RunIf(min_cuda_gpus=1)),
        pytest.param(torch.device("cuda", 0), torch.device("cpu"), marks=RunIf(min_cuda_gpus=1)),
    ],
)
def test_lite_dataloader_device_placement(src_device, dest_device):
    """Test that the LiteDataLoader moves data to the device in its iterator."""
    sample0 = torch.tensor(0, device=src_device)
    sample1 = torch.tensor(1, device=src_device)
    sample2 = {"data": torch.tensor(2, device=src_device)}
    sample3 = {"data": torch.tensor(3, device=src_device)}
    dataloader = DataLoader([sample0, sample1, sample2, sample3], batch_size=2)
    lite_dataloader = _LiteDataLoader(dataloader=dataloader, device=dest_device)
    iterator = iter(lite_dataloader)

    batch0 = next(iterator)
    assert torch.equal(batch0, torch.tensor([0, 1], device=dest_device))

    batch1 = next(iterator)
    assert torch.equal(batch1["data"], torch.tensor([2, 3], device=dest_device))


def test_lite_optimizer_wraps():
    """Test that the LiteOptimizer fully wraps the optimizer."""
    optimizer_cls = torch.optim.SGD
    optimizer = Mock(spec=optimizer_cls)
    lite_optimizer = _LiteOptimizer(optimizer, Mock())
    assert lite_optimizer.optimizer is optimizer
    assert isinstance(lite_optimizer, optimizer_cls)


def test_lite_optimizer_state_dict():
    """Test that the LiteOptimizer calls into the strategy to collect the state."""
    optimizer = Mock()
    strategy = Mock()
    lite_optimizer = _LiteOptimizer(optimizer=optimizer, strategy=strategy)
    lite_optimizer.state_dict()
    strategy.optimizer_state.assert_called_with(optimizer)


def test_lite_optimizer_steps():
    """Test that the LiteOptimizer forwards the step() and zero_grad() calls to the wrapped optimizer."""
    optimizer = Mock()
    strategy = Mock()
    strategy.optimizer_step.return_value = 123
    lite_optimizer = _LiteOptimizer(optimizer=optimizer, strategy=strategy)
    step_output = lite_optimizer.step()
    assert step_output == 123
    strategy.optimizer_step.assert_called_once()
    strategy.optimizer_step.assert_called_with(optimizer, opt_idx=0, closure=ANY, model=strategy.model)
