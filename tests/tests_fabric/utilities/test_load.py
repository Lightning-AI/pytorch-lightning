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

from fabric.utilities.load import _lazy_load, _LazyLoad


def test_lazy_load_module(tmp_path):
    model0 = nn.Linear(2, 2)
    torch.save(model0.state_dict(), tmp_path / "model.pt")

    model1 = nn.Linear(2, 2)
    with _LazyLoad(tmp_path / "model.pt") as checkpoint:
        model1.load_state_dict(checkpoint)

    assert torch.equal(model0.weight, model1.weight)
    assert torch.equal(model0.bias, model1.bias)


def test_lazy_load_tensor(tmp_path):
    data = torch.rand(2, 2)
    torch.save({"data": data}, tmp_path / "data.pt")

    with _LazyLoad(tmp_path / "data.pt") as checkpoint:
        loaded_data = checkpoint["data"]

    assert torch.equal(loaded_data, data)
