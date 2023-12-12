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
import pytest
import torch
import torch.nn as nn
from lightning.fabric.utilities.load import _lazy_load, _materialize_tensors, _move_state_into, _NotYetLoadedTensor

from tests_fabric.helpers.runif import RunIf


@RunIf(min_torch="2.0.0")
def test_lazy_load_module(tmp_path):
    model0 = nn.Linear(2, 2)
    torch.save(model0.state_dict(), tmp_path / "model.pt")

    model1 = nn.Linear(2, 2)
    checkpoint = _lazy_load(tmp_path / "model.pt")
    model1.load_state_dict(checkpoint)

    assert isinstance(checkpoint["weight"], _NotYetLoadedTensor)
    assert type(model0.weight.data) is torch.Tensor
    assert torch.equal(model0.weight, model1.weight)
    assert torch.equal(model0.bias, model1.bias)


class ATensor(torch.Tensor):
    pass


@RunIf(min_torch="2.0.0")
def test_lazy_load_tensor(tmp_path):
    """Test that lazy load can handle different classes of tensors."""
    expected = {
        "tensor": torch.rand(2),
        "parameter": nn.Parameter(torch.rand(3)),
        "subclass": torch.Tensor._make_subclass(ATensor, torch.rand(4)),
    }
    torch.save(expected, tmp_path / "data.pt")

    loaded = _lazy_load(tmp_path / "data.pt")
    for t0, t1 in zip(expected.values(), loaded.values()):
        assert isinstance(t1, _NotYetLoadedTensor)
        t1_materialized = _materialize_tensors(t1)
        assert type(t0) == type(t1_materialized)
        assert torch.equal(t0, t1_materialized)


@RunIf(min_torch="2.0.0")
def test_lazy_load_mixed_state(tmp_path):
    model0 = nn.Linear(2, 2)
    optim0 = torch.optim.Adam(model0.parameters())
    checkpoint = {
        "int": 1,
        "dict": {"a": 1, "b": 2},
        "list": [1, 2, 3],
        "pickled_model": model0,
        "model": model0.state_dict(),
        "optimizer": optim0.state_dict(),
    }
    torch.save(checkpoint, tmp_path / "checkpoint.pt")

    model1 = nn.Linear(2, 2)
    optim1 = torch.optim.Adam(model0.parameters())
    loaded_checkpoint = _lazy_load(tmp_path / "checkpoint.pt")
    model1.load_state_dict(loaded_checkpoint["model"])
    optim1.load_state_dict(loaded_checkpoint["optimizer"])


@RunIf(min_torch="2.0.0")
def test_lazy_load_raises():
    with pytest.raises(FileNotFoundError, match="foo' does not exist"):
        _lazy_load("foo")


@RunIf(min_torch="2.0.0")
def test_materialize_tensors(tmp_path):
    # Single tensor
    tensor = torch.tensor([1, 2])
    torch.save(tensor, tmp_path / "tensor.pt")
    loaded = _lazy_load(tmp_path / "tensor.pt")
    materialized = _materialize_tensors(loaded)
    assert torch.equal(materialized, tensor)
    assert type(tensor) == type(materialized)

    # Collection of tensors
    collection = {
        "tensor": torch.tensor([1, 2]),
        "nested": {"int": 1, "list": [torch.tensor([3.0]), torch.tensor([4])]},
    }
    torch.save(collection, tmp_path / "collection.pt")
    loaded = _lazy_load(tmp_path / "collection.pt")
    materialized = _materialize_tensors(loaded)
    assert torch.equal(materialized["tensor"], collection["tensor"])
    assert torch.equal(materialized["nested"]["list"][0], collection["nested"]["list"][0])
    assert torch.equal(materialized["nested"]["list"][1], collection["nested"]["list"][1])
    assert materialized["nested"]["int"] == 1


def test_move_state_into():
    # all keys from the source
    source = {"apple": 1, "cocofruit": 2}
    destination = {"banana": 100}
    _move_state_into(source, destination)
    assert source == {}
    assert destination == {"apple": 1, "cocofruit": 2, "banana": 100}

    # subset of keys from the source
    source = {"apple": 1, "cocofruit": 2}
    destination = {"banana": 100}
    keys = {"apple"}
    _move_state_into(source, destination, keys=keys)
    assert source == {"cocofruit": 2}
    assert destination == {"apple": 1, "banana": 100}

    # with stateful objects in destination
    class Fruit:
        count = 1

        def state_dict(self):
            return {"count": self.count}

        def load_state_dict(self, state_dict):
            self.count = state_dict["count"]

    source = {"cocofruit": 2, "banana": {"count": 100}}
    destination = {"banana": Fruit()}
    _move_state_into(source, destination)
    assert source == {}
    assert destination["cocofruit"] == 2
    assert destination["banana"].count == 100
