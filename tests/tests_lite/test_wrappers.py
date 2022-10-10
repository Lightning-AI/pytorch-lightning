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
from unittest.mock import Mock

import pytest
import torch
from tests_lite.helpers.runif import RunIf
from torch.utils.data.dataloader import DataLoader

from lightning_lite.lite import LightningLite
from lightning_lite.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_lite.wrappers import _LiteDataLoader, _LiteModule, _LiteOptimizer


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
    """Test that attribute lookup passes through to the original module when possible."""

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


def test_lite_module_state_dict_access():
    """Test that state_dict access passes through to the original module."""

    class OriginalModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 3)

    original_module = OriginalModule()

    class ModuleWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapped = original_module

    wrapped_module = ModuleWrapper()

    lite_module = _LiteModule(wrapped_module, Mock(), original_module=original_module)
    state_dict = lite_module.state_dict()
    assert set(state_dict.keys()) == {"layer.weight", "layer.bias"}

    weight, bias = torch.rand(3, 2), torch.rand(3)
    lite_module.load_state_dict({"layer.weight": weight, "layer.bias": bias})
    assert torch.equal(lite_module.layer.weight, weight)
    assert torch.equal(lite_module.layer.bias, bias)


@pytest.mark.parametrize(
    "precision, input_type, expected_type, accelerator, device_str",
    [
        pytest.param(32, torch.float16, torch.float32, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(32, torch.float32, torch.float32, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(32, torch.float64, torch.float32, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(32, torch.int, torch.int, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(16, torch.float32, torch.float16, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(16, torch.float64, torch.float16, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(16, torch.long, torch.long, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(
            "bf16",
            torch.float32,
            torch.bfloat16,
            "gpu",
            "cuda:0",
            marks=RunIf(min_cuda_gpus=1, min_torch="1.10", bf16_cuda=True),
        ),
        pytest.param(
            "bf16",
            torch.float64,
            torch.bfloat16,
            "gpu",
            "cuda:0",
            marks=RunIf(min_cuda_gpus=1, min_torch="1.10", bf16_cuda=True),
        ),
        pytest.param(
            "bf16",
            torch.bool,
            torch.bool,
            "gpu",
            "cuda:0",
            marks=RunIf(min_cuda_gpus=1, min_torch="1.10", bf16_cuda=True),
        ),
        pytest.param(32, torch.float32, torch.float32, "mps", "mps:0", marks=RunIf(mps=True)),
    ],
)
def test_lite_module_forward_conversion(precision, input_type, expected_type, accelerator, device_str):
    """Test that the LiteModule performs autocasting on the input tensors and during forward()."""
    lite = EmptyLite(precision=precision, accelerator=accelerator, devices=1)
    device = torch.device(device_str)

    def check_autocast(forward_input):
        assert precision != 16 or torch.is_autocast_enabled()
        return forward_input

    module = Mock(wraps=torch.nn.Identity(), side_effect=check_autocast)
    lite_module = _LiteModule(module, lite._precision).to(device)
    out = lite_module(torch.tensor([1, 2, 3], dtype=input_type, device=device))
    assert module.call_args[0][0].dtype == expected_type
    assert out.dtype == input_type or out.dtype == torch.get_default_dtype()


@pytest.mark.parametrize(
    "device_str",
    [
        "cpu",
        pytest.param("cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lite_module_device_dtype_propagation(device_str, dtype):
    """Test that the LiteModule propagates device and dtype properties to its submodules (e.g. torchmetrics)."""

    device = torch.device(device_str)

    class DeviceModule(_DeviceDtypeModuleMixin):
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
    "src_device_str, dest_device_str",
    [
        ("cpu", "cpu"),
        pytest.param("cpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda:0", "cpu", marks=RunIf(min_cuda_gpus=1)),
        # pytest.param("cpu", "mps", marks=RunIf(mps=True)),  # TODO: Add once torch.equal is supported
        pytest.param("mps", "cpu", marks=RunIf(mps=True)),
    ],
)
def test_lite_dataloader_device_placement(src_device_str, dest_device_str):
    """Test that the LiteDataLoader moves data to the device in its iterator."""
    src_device = torch.device(src_device_str)
    dest_device = torch.device(dest_device_str)

    sample0 = torch.tensor(0, device=src_device)
    sample1 = torch.tensor(1, device=src_device)
    sample2 = {"data": torch.tensor(2, device=src_device)}
    sample3 = {"data": torch.tensor(3, device=src_device)}
    dataloader = DataLoader([sample0, sample1, sample2, sample3], batch_size=2)
    lite_dataloader = _LiteDataLoader(dataloader=dataloader, device=dest_device)
    iterator = iter(lite_dataloader)

    batch0 = next(iterator)
    # TODO: torch.equal is not supported on MPS at this time (torch 1.12)
    assert torch.equal(batch0, torch.tensor([0, 1], device=dest_device))

    batch1 = next(iterator)
    # TODO: torch.equal is not supported on MPS at this time (torch 1.12)
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
    strategy.get_optimizer_state.assert_called_with(optimizer)


def test_lite_optimizer_steps():
    """Test that the LiteOptimizer forwards the step() and zero_grad() calls to the wrapped optimizer."""
    optimizer = Mock()
    strategy = Mock(spec=["optimizer_step"])
    strategy.optimizer_step.return_value = 123
    lite_optimizer = _LiteOptimizer(optimizer=optimizer, strategy=strategy)
    step_output = lite_optimizer.step()
    assert step_output == 123
    strategy.optimizer_step.assert_called_once_with(optimizer)

    strategy.reset_mock()

    # with closure as input
    closure = Mock()
    lite_optimizer.step(closure=closure)
    strategy.optimizer_step.assert_called_once_with(optimizer, closure=closure)

    # with model as optimizer
    strategy = Mock(spec=["optimizer_step", "model"])
    lite_optimizer = _LiteOptimizer(optimizer=optimizer, strategy=strategy)
    lite_optimizer.step()
    strategy.optimizer_step.assert_called_once_with(strategy.model)
