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
import pytest
import torch

from lightning_lite.utilities.apply_func import move_data_to_device


@pytest.mark.parametrize("should_return", [False, True])
def test_wrongly_implemented_transferable_data_type(should_return):
    class TensorObject:
        def __init__(self, tensor: torch.Tensor, should_return: bool = True):
            self.tensor = tensor
            self.should_return = should_return

        def to(self, device):
            self.tensor.to(device)
            # simulate a user forgets to return self
            if self.should_return:
                return self

    tensor = torch.tensor(0.1)
    obj = TensorObject(tensor, should_return)
    assert obj == move_data_to_device(obj, torch.device("cpu"))
