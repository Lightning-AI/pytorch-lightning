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
import torch
import torch.nn as nn

from lightning.fabric.utilities.load import _lazy_load, _NotYetLoadedTensor


def test_lazy_load_module(tmp_path):
    model0 = nn.Linear(2, 2)
    torch.save(model0.state_dict(), tmp_path / "model.pt")

    model1 = nn.Linear(2, 2)
    with _lazy_load(tmp_path / "model.pt") as checkpoint:
        model1.load_state_dict(checkpoint)

    assert isinstance(checkpoint["weight"], _NotYetLoadedTensor)
    assert type(model0.weight.data) == torch.Tensor
    assert torch.equal(model0.weight, model1.weight)
    assert torch.equal(model0.bias, model1.bias)


class ATensor(torch.Tensor):
    pass


def test_lazy_load_tensor(tmp_path):
    """Test that lazy load can handle different classes of tensors."""
    expected = {
        "tensor": torch.rand(2),
        "parameter": nn.Parameter(torch.rand(3)),
        "subclass": torch.Tensor._make_subclass(ATensor, torch.rand(4)),
    }
    torch.save(expected, tmp_path / "data.pt")

    with _lazy_load(tmp_path / "data.pt") as loaded:
        for t0, t1 in zip(expected.values(), loaded.values()):
            assert isinstance(t1, _NotYetLoadedTensor)
            t1_materialized = t1._load_tensor()
            assert type(t0) == type(t1_materialized)
            assert torch.equal(t0, t1_materialized)


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
    with _lazy_load(tmp_path / "checkpoint.pt") as loaded_checkpoint:
        model1.load_state_dict(loaded_checkpoint["model"])
        optim1.load_state_dict(loaded_checkpoint["optimizer"])
