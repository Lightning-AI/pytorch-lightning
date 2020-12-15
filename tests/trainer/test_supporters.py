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

from pytorch_lightning.trainer.supporters import TensorRunningAccum


def test_tensor_running_accum_reset():
    """ Test that reset would set all attributes to the initialization state """

    window_length = 10

    accum = TensorRunningAccum(window_length=window_length)
    assert accum.last() is None
    assert accum.mean() is None

    accum.append(torch.tensor(1.5))
    assert accum.last() == torch.tensor(1.5)
    assert accum.mean() == torch.tensor(1.5)

    accum.reset()
    assert accum.window_length == window_length
    assert accum.memory is None
    assert accum.current_idx == 0
    assert accum.last_idx is None
    assert not accum.rotated
