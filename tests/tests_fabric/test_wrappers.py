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
from unittest.mock import Mock, call

import pytest
import torch
from lightning.fabric.fabric import Fabric
from lightning.fabric.plugins import Precision
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.wrappers import (
    _FabricDataLoader,
    _FabricModule,
    _FabricOptimizer,
    _unwrap_compiled,
    _unwrap_objects,
    is_wrapped,
)
from torch._dynamo import OptimizedModule
from torch.utils.data import BatchSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader

from tests_fabric.helpers.runif import RunIf


def test_fabric_module_wraps():
    """Test that the wrapped module is accessible via the property."""
    module = Mock()
    assert _FabricModule(module, Mock()).module is module

    wrapped_module = Mock()
    original_module = Mock()
    assert _FabricModule(wrapped_module, Mock(), original_module=original_module).module is original_module


def test_fabric_module_attribute_lookup():
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

    fabric_module = _FabricModule(wrapped_module, Mock(), original_module=original_module)
    assert fabric_module.attribute == 1
    assert fabric_module.layer is original_module.layer
    assert fabric_module.forward.__self__.__class__ == _FabricModule

    with pytest.raises(AttributeError):
        _ = fabric_module.not_exists


def test_fabric_module_method_lookup():
    """Test that access to methods warns about improper use when a wrapper from a strategy is involved."""

    class OriginalModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule = torch.nn.Linear(2, 3)

        def forward(self, x):
            return x

        def method_without_module_invocation(self):
            return 100

        def method_with_submodule_invocation(self):
            self.submodule(torch.rand(2, 2))
            return 101

        def method_with_self_invocation(self):
            self(None)
            return 102

    class ModuleWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.wrapped = module

        def forward(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

    # Regular case: forward_module == original_module -> no warnings
    original_module = OriginalModule()
    fabric_module = _FabricModule(forward_module=original_module, strategy=Mock(), original_module=original_module)
    assert fabric_module.method_without_module_invocation() == 100

    # Special case: original module wrapped by forward module: -> error if method requires rerouting
    original_module = OriginalModule()
    wrapped_module = ModuleWrapper(original_module)
    fabric_module = _FabricModule(
        forward_module=wrapped_module, strategy=Mock(precision=Precision()), original_module=original_module
    )
    assert fabric_module.method_without_module_invocation() == 100
    with pytest.raises(
        RuntimeError, match=r"You are calling the method `OriginalModule.method_with_submodule_invocation\(\)` from"
    ):
        assert fabric_module.method_with_submodule_invocation() == 101
    with pytest.raises(
        RuntimeError, match=r"You are calling the method `OriginalModule.method_with_self_invocation\(\)` from"
    ):
        assert fabric_module.method_with_self_invocation() == 102

    # No error if explicitly marked as forward method
    fabric_module.mark_forward_method("method_with_self_invocation")
    assert fabric_module.method_with_self_invocation() == 102


def test_fabric_module_mark_forward_method():
    class OriginalModule(torch.nn.Module):
        attribute = 1

        def forward(self, x):
            return x

        def special(self):
            pass

    original_module = OriginalModule()
    fabric_module = _FabricModule(original_module, Mock(), original_module=original_module)

    with pytest.raises(ValueError, match="You cannot mark the forward method itself"):
        fabric_module.mark_forward_method("forward")

    with pytest.raises(AttributeError, match="`OriginalModule.not_exist` does not exist or is not a method."):
        fabric_module.mark_forward_method("not_exist")

    with pytest.raises(AttributeError, match="`OriginalModule.attribute` does not exist or is not a method."):
        fabric_module.mark_forward_method("attribute")

    def special(x):
        return x

    with pytest.raises(TypeError, match="Expected a method or a string"):
        fabric_module.mark_forward_method(special)

    lightning_module_methods = {"training_step", "validation_step", "test_step", "predict_step"}
    assert fabric_module._forward_methods == lightning_module_methods

    # Mark via name
    fabric_module.mark_forward_method("special")
    assert fabric_module._forward_methods == {"special"} | lightning_module_methods

    # Mark by passing in the method itself
    fabric_module = _FabricModule(original_module, Mock(), original_module=original_module)
    fabric_module.mark_forward_method(original_module.special)
    assert fabric_module._forward_methods == {"special"} | lightning_module_methods


def test_fabric_module_setattr():
    """Test that setattr sets attributes on the original module."""

    class OriginalModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 3)
            self.attribute = 1
            self._x = None

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    original_module = OriginalModule()

    class ModuleWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapped = original_module

    wrapped_module = ModuleWrapper()
    fabric_module = _FabricModule(wrapped_module, Mock(), original_module=original_module)

    # Check new attribute is set on original_module
    fabric_module.new_attribute = 100
    assert original_module.new_attribute == 100

    # Modify existing attribute on original_module
    fabric_module.attribute = 101
    # "attribute" is only in the original_module, so it shouldn't get set in the fabric_module
    assert "attribute" not in fabric_module.__dict__
    assert fabric_module.attribute == 101  # returns it from original_module
    assert original_module.attribute == 101

    # Check setattr of original_module
    fabric_module.x = 102
    assert original_module.x == 102

    # Check set submodule
    assert not hasattr(original_module, "linear")
    linear = torch.nn.Linear(2, 2)
    fabric_module.linear = linear
    assert hasattr(original_module, "linear")
    assert isinstance(original_module.linear, torch.nn.Module)
    assert linear in fabric_module.modules()
    assert linear in original_module.modules()

    # Check monkeypatching of methods
    fabric_module = _FabricModule(Mock(), Mock())
    original = id(fabric_module.forward)
    fabric_module.forward = lambda *_: None
    assert id(fabric_module.forward) != original
    # Check special methods
    assert "__repr__" in dir(fabric_module)
    assert "__repr__" not in fabric_module.__dict__
    assert "__repr__" not in _FabricModule.__dict__
    fabric_module.__repr__ = lambda *_: "test"
    assert fabric_module.__repr__() == "test"
    # needs to be monkeypatched on the class for `repr()` to change
    assert repr(fabric_module) == "_FabricModule()"
    with mock.patch.object(_FabricModule, "__repr__", return_value="test"):
        assert fabric_module.__repr__() == "test"
        assert repr(fabric_module) == "test"


def test_fabric_module_state_dict_access():
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

    fabric_module = _FabricModule(wrapped_module, Mock(), original_module=original_module)
    state_dict = fabric_module.state_dict()
    assert set(state_dict.keys()) == {"layer.weight", "layer.bias"}

    weight, bias = torch.rand(3, 2), torch.rand(3)
    fabric_module.load_state_dict({"layer.weight": weight, "layer.bias": bias})
    assert torch.equal(fabric_module.layer.weight, weight)
    assert torch.equal(fabric_module.layer.bias, bias)

    if _TORCH_GREATER_EQUAL_2_1:
        # Can use additional `assign` argument in PyTorch >= 2.1
        with torch.device("meta"):
            original_module = OriginalModule()
        fabric_module = _FabricModule(wrapped_module, Mock(), original_module=original_module)
        assert fabric_module.layer.weight.is_meta
        fabric_module.load_state_dict({"layer.weight": weight, "layer.bias": bias}, assign=True)
        assert not fabric_module.layer.weight.is_meta


@pytest.mark.parametrize(
    ("precision", "input_type", "expected_type", "accelerator", "device_str"),
    [
        pytest.param(32, torch.float16, torch.float16, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(32, torch.float32, torch.float32, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param(32, torch.float64, torch.float64, "gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
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
            marks=RunIf(min_cuda_gpus=1, bf16_cuda=True),
        ),
        pytest.param(
            "bf16",
            torch.float64,
            torch.bfloat16,
            "gpu",
            "cuda:0",
            marks=RunIf(min_cuda_gpus=1, bf16_cuda=True),
        ),
        pytest.param(
            "bf16",
            torch.bool,
            torch.bool,
            "gpu",
            "cuda:0",
            marks=RunIf(min_cuda_gpus=1, bf16_cuda=True),
        ),
        pytest.param(32, torch.float32, torch.float32, "mps", "mps:0", marks=RunIf(mps=True)),
    ],
)
def test_fabric_module_forward_conversion(precision, input_type, expected_type, accelerator, device_str):
    """Test that the FabricModule performs autocasting on the input tensors and during forward()."""
    fabric = Fabric(precision=precision, accelerator=accelerator, devices=1)
    device = torch.device(device_str)

    def check_autocast(forward_input):
        assert precision != 16 or torch.is_autocast_enabled()
        return forward_input

    module = Mock(wraps=torch.nn.Identity(), side_effect=check_autocast)
    fabric_module = _FabricModule(module, fabric._strategy).to(device)
    out = fabric_module(torch.tensor([1, 2, 3], dtype=input_type, device=device))
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
def test_fabric_module_device_dtype_propagation(device_str, dtype):
    """Test that the FabricModule propagates device and dtype properties to its submodules (e.g. torchmetrics)."""
    device = torch.device(device_str)

    class DeviceModule(_DeviceDtypeModuleMixin):
        pass

    device_module = DeviceModule()
    fabric_module = _FabricModule(device_module, Mock())
    fabric_module.to(device)
    assert device_module.device == device
    assert fabric_module.device == device

    fabric_module.to(dtype)
    assert device_module.dtype == dtype
    assert fabric_module.dtype == dtype


def test_fabric_dataloader_iterator():
    """Test that the iteration over a FabricDataLoader wraps the iterator of the underlying dataloader (no automatic
    device placement)."""
    dataloader = DataLoader(range(5), batch_size=2)
    fabric_dataloader = _FabricDataLoader(dataloader)
    assert len(fabric_dataloader) == len(dataloader) == 3

    iterator = iter(dataloader)
    fabric_iterator = iter(fabric_dataloader)

    assert torch.equal(next(iterator), next(fabric_iterator))
    assert torch.equal(next(iterator), next(fabric_iterator))
    assert torch.equal(next(iterator), next(fabric_iterator))

    with pytest.raises(StopIteration):
        next(iterator)

    with pytest.raises(StopIteration):
        next(fabric_iterator)


@pytest.mark.parametrize(
    ("src_device_str", "dest_device_str"),
    [
        ("cpu", "cpu"),
        pytest.param("cpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda:0", "cpu", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cpu", "mps", marks=RunIf(mps=True)),
        pytest.param("mps", "cpu", marks=RunIf(mps=True)),
    ],
)
def test_fabric_dataloader_device_placement(src_device_str, dest_device_str):
    """Test that the FabricDataLoader moves data to the device in its iterator."""
    src_device = torch.device(src_device_str)
    dest_device = torch.device(dest_device_str)

    sample0 = torch.tensor(0, device=src_device)
    sample1 = torch.tensor(1, device=src_device)
    sample2 = {"data": torch.tensor(2, device=src_device)}
    sample3 = {"data": torch.tensor(3, device=src_device)}
    dataloader = DataLoader([sample0, sample1, sample2, sample3], batch_size=2)
    fabric_dataloader = _FabricDataLoader(dataloader=dataloader, device=dest_device)
    iterator = iter(fabric_dataloader)

    batch0 = next(iterator)
    assert torch.equal(batch0, torch.tensor([0, 1], device=dest_device))

    batch1 = next(iterator)
    assert torch.equal(batch1["data"], torch.tensor([2, 3], device=dest_device))


@pytest.mark.parametrize("use_batch_sampler", [False, True])
def test_fabric_dataloader_distributed_sampler_set_epoch(use_batch_sampler):
    """Test that the FabricDataLoader calls `set_epoch()` on the wrapped sampler if applicable."""
    dataset = range(3)
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0)
    sampler.set_epoch = Mock()

    if not use_batch_sampler:
        dataloader = DataLoader(dataset, sampler=sampler)
    else:
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    fabric_dataloader = _FabricDataLoader(dataloader)
    iterator_epoch_0 = iter(fabric_dataloader)
    sampler.set_epoch.assert_not_called()

    next(iterator_epoch_0)
    # .set_epoch() gets called before the first sample gets fetched from the wrapped dataloader
    assert sampler.set_epoch.mock_calls == [call(0)]

    next(iterator_epoch_0)
    assert sampler.set_epoch.mock_calls == [call(0)]

    iterator_epoch_1 = iter(fabric_dataloader)
    assert sampler.set_epoch.mock_calls == [call(0)]

    next(iterator_epoch_1)
    # with every new iterator call, the epoch increases
    assert sampler.set_epoch.mock_calls == [call(0), call(1)]


def test_fabric_optimizer_wraps():
    """Test that the FabricOptimizer fully wraps the optimizer."""
    optimizer_cls = torch.optim.SGD
    optimizer = Mock(spec=optimizer_cls)
    fabric_optimizer = _FabricOptimizer(optimizer, Mock())
    assert fabric_optimizer.optimizer is optimizer
    assert isinstance(fabric_optimizer, optimizer_cls)
    assert isinstance(fabric_optimizer, _FabricOptimizer)
    assert type(fabric_optimizer).__name__ == "FabricSGD"


def test_fabric_optimizer_state_dict():
    """Test that the FabricOptimizer calls into the strategy to collect the state."""
    optimizer = Mock(spec=torch.optim.Adam)
    strategy = Mock()
    fabric_optimizer = _FabricOptimizer(optimizer=optimizer, strategy=strategy)
    fabric_optimizer.state_dict()
    strategy.get_optimizer_state.assert_called_with(optimizer)


def test_fabric_optimizer_load_state_dict():
    """Test that the FabricOptimizer can load the state dict on the wrapped optimizer and update its internal
    `__dict__`."""
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    assert not optimizer.state  # a fresh optimizer has no state
    model(torch.rand(1)).backward()
    optimizer.step()
    assert optimizer.state
    state_dict = optimizer.state_dict()

    optimizer = torch.optim.Adam(model.parameters())  # fresh optimizer
    fabric_optimizer = _FabricOptimizer(optimizer=optimizer, strategy=Mock())
    assert not fabric_optimizer.state  # a fresh optimizer has no state
    fabric_optimizer.load_state_dict(state_dict)
    assert fabric_optimizer.state
    assert fabric_optimizer.optimizer.state_dict()["state"] == state_dict["state"]


def test_fabric_optimizer_steps():
    """Test that the FabricOptimizer forwards the step() and zero_grad() calls to the wrapped optimizer."""
    optimizer = Mock(spec=torch.optim.Adam)
    strategy = Mock(spec=["optimizer_step"])
    strategy.optimizer_step.return_value = 123
    fabric_optimizer = _FabricOptimizer(optimizer=optimizer, strategy=strategy)
    step_output = fabric_optimizer.step()
    assert step_output == 123
    strategy.optimizer_step.assert_called_once_with(optimizer)

    strategy.reset_mock()

    # with closure as input
    closure = Mock()
    fabric_optimizer.step(closure=closure)
    strategy.optimizer_step.assert_called_once_with(optimizer, closure=closure)

    # with model as optimizer
    strategy = Mock(spec=["optimizer_step", "model"])
    fabric_optimizer = _FabricOptimizer(optimizer=optimizer, strategy=strategy)
    fabric_optimizer.step()
    strategy.optimizer_step.assert_called_once_with(strategy.model)


def test_fabric_optimizer_zero_grad_kwargs():
    """Test that Fabric can adapt the `.zero_grad()` arguments to the underlying optimizer."""
    # Test PyTorch's standard `.zero_grad()` signature
    with mock.patch("torch.optim.SGD.zero_grad") as zero_grad_mock:
        optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), 0.1)
        fabric_optimizer = _FabricOptimizer(optimizer=optimizer, strategy=Mock())
        fabric_optimizer.zero_grad()
        zero_grad_mock.assert_called_with()
        fabric_optimizer.zero_grad(set_to_none=False)
        zero_grad_mock.assert_called_with(set_to_none=False)
        fabric_optimizer.zero_grad(set_to_none=True)
        zero_grad_mock.assert_called_with(set_to_none=True)

    # Test weird `.zero_grad()` signatures from other libraries
    custom_zero_grad = Mock()

    class CustomSGD(torch.optim.SGD):
        def zero_grad(self, set_grads_to_None=False):
            custom_zero_grad(set_grads_to_None=set_grads_to_None)

    optimizer = CustomSGD(torch.nn.Linear(1, 1).parameters(), 0.1)
    fabric_optimizer = _FabricOptimizer(optimizer=optimizer, strategy=Mock())
    fabric_optimizer.zero_grad()
    custom_zero_grad.assert_called_with(set_grads_to_None=False)


@pytest.mark.parametrize("compile", [False, pytest.param(True, marks=RunIf(dynamo=True))])
def test_is_wrapped(compile):
    """Test that the `is_wrapped` utility recognizes when an object was wrapped by Fabric."""
    assert not is_wrapped(None)

    # _FabricModule
    module = torch.nn.Linear(2, 2)
    assert not is_wrapped(module)
    wrapped = _FabricModule(module, Mock())
    assert is_wrapped(wrapped)

    # _FabricModule inside an OptimizedModule
    if compile:
        module = torch.nn.Linear(2, 2)
        wrapped = torch.compile(_FabricModule(module, Mock()))
        assert isinstance(wrapped, OptimizedModule)
        assert is_wrapped(wrapped)

    # _FabricOptimizer
    optimizer = torch.optim.Adam(module.parameters())
    assert not is_wrapped(optimizer)
    wrapped = _FabricOptimizer(optimizer, Mock())
    assert is_wrapped(wrapped)

    # _FabricDataLoader
    dataloader = DataLoader([1, 2, 3])
    assert not is_wrapped(dataloader)
    wrapped = _FabricDataLoader(dataloader)
    assert is_wrapped(wrapped)


@pytest.mark.parametrize("compile", [False, pytest.param(True, marks=RunIf(dynamo=True))])
def test_unwrap_objects(compile):
    # empty container
    assert _unwrap_objects({}) == {}

    # container with pure objects and wrapped objects
    module = torch.nn.Linear(1, 1)
    wrapped_module = _FabricModule(module, Mock())
    if compile:
        wrapped_module = torch.compile(wrapped_module)
    optimizer = torch.optim.Adam(module.parameters())
    wrapped_optimizer = _FabricOptimizer(optimizer, Mock())
    dataloader = DataLoader([1, 2, 3])
    wrapped_dataloader = _FabricDataLoader(dataloader)
    container = {
        "int": 1,
        "module": module,
        "wrapped_module": wrapped_module,
        "optimizer": optimizer,
        "wrapped_optimizer": wrapped_optimizer,
        "dataloader": dataloader,
        "wrapped_dataloader": wrapped_dataloader,
        "nested": [module, wrapped_module, optimizer, wrapped_optimizer, dataloader, wrapped_dataloader],
    }
    expected = {
        "int": 1,
        "module": module,
        "wrapped_module": wrapped_module._forward_module,
        "optimizer": optimizer,
        "wrapped_optimizer": optimizer,
        "dataloader": dataloader,
        "wrapped_dataloader": dataloader,
        "nested": [module, wrapped_module._forward_module, optimizer, optimizer, dataloader, dataloader],
    }
    assert _unwrap_objects(container) == expected


def test_step_method_redirection():
    """Test that the FabricModule redirects methods marked as 'forward methods' through forward to avoid bypassing the
    DDP/FSDP wrappers."""

    class DDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class LightningModule(torch.nn.Module):
        def forward(self):
            return "forward_return"

        def training_step(self, arg, kwarg=None):
            assert self() == "forward_return"
            assert arg == "train_arg"
            assert kwarg == "train_kwarg"
            return "training_step_return"

        def marked_method(self, arg, kwarg=None):
            assert self() == "forward_return"
            assert arg == "marked_arg"
            assert kwarg == "marked_kwarg"
            return "marked_method_return"

        def normal_method(self):
            pass

    strategy = Mock()
    strategy.precision = Mock(wraps=Precision())
    original_module = LightningModule()
    forward_module = DDP(original_module)
    fabric_module = _FabricModule(forward_module=forward_module, strategy=strategy, original_module=original_module)

    # Regular methods on the original_module are visible and identical on the fabric_module ...
    assert fabric_module.normal_method.__wrapped__ == original_module.normal_method

    # ... but special methods like training_step get redirected to the forward_module
    assert fabric_module.training_step.__name__ == "call_forward_module"
    assert fabric_module.validation_step.__name__ == "call_forward_module"
    assert fabric_module.test_step.__name__ == "call_forward_module"
    assert fabric_module.predict_step.__name__ == "call_forward_module"

    with pytest.raises(AttributeError, match="has no attribute 'predict_step'"):
        # A special method that does not exist will raise its AttributeError when being called
        fabric_module.predict_step()

    # The forward method on the original module remains untouched
    assert original_module.forward.__name__ == "forward"

    # The special methods get redirected correctly to produce the expected output
    strategy.precision.forward_context.reset_mock()
    assert fabric_module.training_step("train_arg", kwarg="train_kwarg") == "training_step_return"
    assert fabric_module.training_step("train_arg", kwarg="train_kwarg") == "training_step_return"  # call 2nd time
    assert strategy.precision.forward_context.call_count == 2

    # Other methods must be marked explicitly to be redirected
    strategy.precision.forward_context.reset_mock()
    with pytest.raises(RuntimeError, match="You are calling the method .* from outside the model"):
        fabric_module.marked_method("marked_arg", kwarg="marked_kwarg")
    fabric_module.mark_forward_method("marked_method")
    assert fabric_module.marked_method("marked_arg", kwarg="marked_kwarg") == "marked_method_return"
    strategy.precision.forward_context.assert_called_once()

    # The forward method remains untouched/unpatched after the special methods have been called
    assert original_module.forward.__name__ == "forward"

    # Special case: forward_module == original_module -> no special treatment applied
    fabric_module = _FabricModule(forward_module=original_module, strategy=Mock(), original_module=original_module)
    assert fabric_module.training_step == original_module.training_step
    assert fabric_module.marked_method == original_module.marked_method


@RunIf(dynamo=True)
def test_unwrap_compiled():
    model = torch.nn.Linear(1, 1)

    # We wrap `torch.compile` on import of lightning in `wrappers.py`
    assert torch.compile.__wrapped__

    compiled = torch.compile(model, fullgraph=True, dynamic=True, disable=False)
    assert compiled._compile_kwargs == {"fullgraph": True, "dynamic": True, "disable": False}
    unwrapped, compile_kwargs = _unwrap_compiled(compiled)
    assert unwrapped is compiled._orig_mod
    assert compile_kwargs == {"fullgraph": True, "dynamic": True, "disable": False}

    del compiled._compile_kwargs
    with pytest.raises(RuntimeError, match="Failed to determine the arguments that were used to compile the module"):
        _unwrap_compiled(compiled)

    # can still be applied as decorator
    @torch.compile()
    def cos(x):
        return torch.cos(x)

    @torch.compile
    def sin(x):
        return torch.sin(x)
