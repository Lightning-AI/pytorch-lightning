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
from unittest.mock import Mock, PropertyMock

import pytest
import torch
from lightning.fabric.plugins import DoublePrecision, HalfPrecision, Precision
from lightning.fabric.strategies import SingleDeviceStrategy
from lightning.fabric.utilities.types import _Stateful

from tests_fabric.helpers.runif import RunIf


@pytest.mark.parametrize("is_rank_zero", [True, False])
def test_save_checkpoint_rank_zero_only(is_rank_zero, tmp_path):
    """Test that the checkpoint only gets saved on global rank 0 in the base implementation in Strategy."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    save_checkpoint_mock = Mock()
    strategy.checkpoint_io.save_checkpoint = save_checkpoint_mock
    with mock.patch(
        "lightning.fabric.strategies.single_device.SingleDeviceStrategy.is_global_zero",
        new_callable=PropertyMock(return_value=is_rank_zero),
    ):
        strategy.save_checkpoint(tmp_path, {"anything": 1})
    assert save_checkpoint_mock.call_count == int(is_rank_zero)


def test_save_checkpoint_empty_state(tmp_path):
    """Test that one can save an empty state with the base implementation in Strategy."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    save_checkpoint_mock = Mock()
    strategy.checkpoint_io.save_checkpoint = save_checkpoint_mock

    state = {}
    strategy.save_checkpoint(tmp_path, state)
    save_checkpoint_mock.assert_called_with(checkpoint=state, path=tmp_path, storage_options=None)


def test_save_checkpoint_convert_stateful_objects(tmp_path):
    """Test that when modules and optimizers are at the top-level in the state, their `state_dict()` gets used."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    save_checkpoint_mock = Mock()
    strategy.checkpoint_io.save_checkpoint = save_checkpoint_mock

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    anything = {"cocofruit": 1}
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "anything": anything}
    expected = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "anything": anything,
    }
    strategy.save_checkpoint(tmp_path, state)
    assert save_checkpoint_mock.call_args[1]["checkpoint"].keys() == expected.keys()
    saved_model_state = save_checkpoint_mock.call_args[1]["checkpoint"]["model"]
    assert all(torch.equal(p0, p1) for p0, p1 in zip(saved_model_state.values(), expected["model"].values()))
    assert save_checkpoint_mock.call_args[1]["checkpoint"]["optimizer"] == expected["optimizer"]
    assert save_checkpoint_mock.call_args[1]["checkpoint"]["scheduler"] == expected["scheduler"]
    assert save_checkpoint_mock.call_args[1]["checkpoint"]["anything"] == expected["anything"]


def test_save_load_stateful_objects(tmp_path):
    """Test that stateful objects other than modules and optimizers get converted and loaded correctly."""

    class Fruit:
        count = 1

        def state_dict(self):
            return {"count": self.count}

        def load_state_dict(self, state_dict):
            self.count = state_dict["count"]

    state = Fruit()
    state.count = 100
    assert isinstance(state, _Stateful)
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    strategy.save_checkpoint(tmp_path / "checkpoint.ckpt", {"state": state})
    state = Fruit()
    assert state.count == 1
    strategy.load_checkpoint(tmp_path / "checkpoint.ckpt", {"state": state})
    assert state.count == 100


def test_load_module_state_dict():
    """Test that `Strategy.load_module_state_dict()` calls `.load_state_dict()` on the module."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    module = Mock()
    state_dict = Mock()
    strategy.load_module_state_dict(module, state_dict)
    module.load_state_dict.assert_called_with(state_dict, strict=True)
    strategy.load_module_state_dict(module, state_dict, strict=False)
    module.load_state_dict.assert_called_with(state_dict, strict=False)


def test_load_checkpoint_model_optimizer_from_raw_checkpoint(tmp_path):
    """Test that the `load_checkpoint` can load raw state dict checkpoints too."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    torch.save(model.state_dict(), tmp_path / "model.ckpt")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.ckpt")

    new_model = torch.nn.Linear(3, 3)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=2.0)

    strategy.load_checkpoint(tmp_path / "model.ckpt", state=new_model, strict=False)
    assert torch.equal(new_model.weight, model.weight)
    strategy.load_checkpoint(tmp_path / "optimizer.ckpt", state=new_optimizer, strict=False)
    assert new_optimizer.state_dict()["param_groups"][0]["lr"] == 1.0


def test_load_checkpoint_out_of_place(tmp_path):
    """Test that one can load the full checkpoint into memory just like `torch.load()`."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    load_checkpoint_mock = Mock()
    strategy.checkpoint_io.load_checkpoint = load_checkpoint_mock

    checkpoint = strategy.load_checkpoint(tmp_path, state=None)
    assert checkpoint == load_checkpoint_mock()

    checkpoint = strategy.load_checkpoint(tmp_path, state={})
    assert checkpoint == load_checkpoint_mock()


def test_load_checkpoint_in_place(tmp_path):
    """Test that the object's state gets reloaded in-place."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class

    # objects with initial state
    saved_model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.Adam(saved_model.parameters(), lr=0.1)
    saved_state = {"model": saved_model, "optimizer": saved_optimizer, "int": 1, "dict": {"cocofruit": 2}}
    strategy.save_checkpoint(tmp_path / "checkpoint", state=saved_state)

    # same objects with different state
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
    state = {"model": model, "optimizer": optimizer, "int": 10, "dict": {"cocofruit": 20}}
    assert not torch.equal(model.weight, saved_model.weight)
    assert optimizer.state_dict() != saved_optimizer.state_dict()

    remainder = strategy.load_checkpoint(tmp_path / "checkpoint", state)
    assert torch.equal(model.weight, saved_model.weight)
    assert optimizer.state_dict() == saved_optimizer.state_dict()
    assert state["int"] == saved_state["int"]
    assert state["dict"] == saved_state["dict"]
    assert not remainder

    # partial load - only model, no optimizer
    model = torch.nn.Linear(2, 2)
    state = {"model": model}
    remainder = strategy.load_checkpoint(tmp_path / "checkpoint", state)
    assert torch.equal(model.weight, saved_model.weight)
    assert list(remainder.keys()) == ["optimizer", "int", "dict"]


def test_load_checkpoint_strict_loading(tmp_path):
    """Test that an error is raised if a key is requested to be restored but does not exist in the checkpoint."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class
    saved_state = {"a": 1, "b": 2}
    requested_state = {"a": 1, "b": 2, "c": 3}  # key `c` does not exist in the saved state
    load_checkpoint_mock = Mock(return_value=saved_state)
    strategy.checkpoint_io.load_checkpoint = load_checkpoint_mock
    with pytest.raises(KeyError, match="contains a key 'c' that does not exist"):
        strategy.load_checkpoint(tmp_path, requested_state, strict=True)


def test_load_checkpoint_non_strict_loading(tmp_path):
    """Test that no error is raised if `strict=False` and state is requested that does not exist in the checkpoint."""
    strategy = SingleDeviceStrategy()  # surrogate class to test implementation in base class

    # objects with initial state
    saved_model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.Adam(saved_model.parameters(), lr=0.1)
    saved_state = {"model": saved_model, "optimizer": saved_optimizer, "int": 1, "str": "test"}
    strategy.save_checkpoint(tmp_path / "checkpoint.ckpt", state=saved_state)

    # same objects with different state
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
    state = {"model": model, "optimizer": optimizer, "int": 2, "new": "not_present_in_saved_state"}
    assert not torch.equal(model.weight, saved_model.weight)
    assert optimizer.state_dict() != saved_optimizer.state_dict()

    remainder = strategy.load_checkpoint(tmp_path / "checkpoint.ckpt", state, strict=False)
    assert torch.equal(model.weight, saved_model.weight)
    assert optimizer.state_dict() == saved_optimizer.state_dict()
    assert state["int"] == saved_state["int"]
    assert "str" not in state
    assert "str" in remainder
    assert state["new"] == "not_present_in_saved_state"
    assert "new" not in remainder


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps:0", marks=RunIf(mps=True)),
    ],
)
@pytest.mark.parametrize(
    ("precision", "dtype"),
    [
        (Precision(), torch.float32),
        (HalfPrecision("16-true"), torch.float16),
        pytest.param(HalfPrecision("bf16-true"), torch.bfloat16, marks=RunIf(mps=False)),
        pytest.param(DoublePrecision(), torch.float64, marks=RunIf(mps=False)),
    ],
)
@pytest.mark.parametrize("empty_init", [None, True, False])
def test_module_init_context(device, precision, dtype, empty_init, monkeypatch):
    """Test that the module under the init-module-context gets moved to the right device and dtype."""
    init_mock = Mock()
    monkeypatch.setattr(torch.Tensor, "uniform_", init_mock)

    device = torch.device(device)
    strategy = SingleDeviceStrategy(device=device, precision=precision)  # surrogate class to test base class
    with strategy.module_init_context(empty_init=empty_init):
        module = torch.nn.Linear(2, 2)

    assert module.weight.device == module.bias.device == device
    assert module.weight.dtype == module.bias.dtype == dtype
    if not empty_init:
        init_mock.assert_called()
    else:
        init_mock.assert_not_called()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps:0", marks=RunIf(mps=True)),
    ],
)
@pytest.mark.parametrize(
    ("precision", "dtype"),
    [
        (Precision(), torch.float32),
        (HalfPrecision("16-true"), torch.float16),
        pytest.param(HalfPrecision("bf16-true"), torch.bfloat16, marks=RunIf(mps=False)),
        pytest.param(DoublePrecision(), torch.float64, marks=RunIf(mps=False)),
    ],
)
def test_tensor_init_context(device, precision, dtype):
    """Test that tensors under the init-tensor-context get moved to the right device and dtype."""
    device = torch.device(device)
    strategy = SingleDeviceStrategy(device=device, precision=precision)  # surrogate class to test base class
    with strategy.tensor_init_context():
        tensor0 = torch.tensor(42.0)
        tensor1 = torch.tensor(42)
        tensor2 = torch.tensor(42.0, dtype=torch.half)

    assert tensor0.device == tensor1.device == tensor2.device == device
    assert tensor0.dtype == dtype
    assert tensor1.dtype == torch.long  # `.init_tensor()` only affects floating point dtypes
    assert tensor2.dtype == torch.half  # this tensor was created with an explicit dtype assignment
