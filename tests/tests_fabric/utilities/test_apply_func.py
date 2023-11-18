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
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, move_data_to_device
from torch import Tensor


@pytest.mark.parametrize("should_return", [False, True])
def test_wrongly_implemented_transferable_data_type(should_return):
    class TensorObject:
        def __init__(self, tensor: Tensor, should_return: bool = True):
            self.tensor = tensor
            self.should_return = should_return

        def to(self, device):
            self.tensor.to(device)
            # simulate a user forgets to return self
            if self.should_return:
                return self
            return None

    tensor = torch.tensor(0.1)
    obj = TensorObject(tensor, should_return)
    assert obj == move_data_to_device(obj, torch.device("cpu"))


def test_convert_tensors_to_scalars():
    assert convert_tensors_to_scalars("string") == "string"
    assert convert_tensors_to_scalars(1) == 1
    assert convert_tensors_to_scalars(True) is True
    assert convert_tensors_to_scalars({"scalar": 1.0}) == {"scalar": 1.0}

    result = convert_tensors_to_scalars({"tensor": torch.tensor(2.0)})
    # note: `==` comparison as above is not sufficient, since `torch.tensor(x) == x` evaluates to truth
    assert not isinstance(result["tensor"], Tensor)
    assert result["tensor"] == 2.0

    data = {"tensor": torch.tensor([2.0])}
    result = convert_tensors_to_scalars(data)
    assert not isinstance(result["tensor"], Tensor)
    assert result["tensor"] == 2.0
    assert isinstance(data["tensor"], Tensor)
    assert data["tensor"] == 2.0

    with pytest.raises(ValueError, match="does not contain a single element"):
        convert_tensors_to_scalars({"tensor": torch.tensor([1, 2, 3])})
