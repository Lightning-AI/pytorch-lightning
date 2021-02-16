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
import torch

from pytorch_lightning.utilities.memory import recursive_detach


def test_recursive_detach():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = {"foo": torch.tensor(0, device=device), "bar": {"baz": torch.tensor(1.0, device=device, requires_grad=True)}}
    y = recursive_detach(x, to_cpu=True)

    assert x["foo"].device.type == device
    assert x["bar"]["baz"].device.type == device
    assert x["bar"]["baz"].requires_grad

    assert y["foo"].device.type == "cpu"
    assert y["bar"]["baz"].device.type == "cpu"
    assert not y["bar"]["baz"].requires_grad
