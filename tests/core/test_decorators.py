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

from pytorch_lightning.core.decorators import auto_move_data
from tests.helpers import BoringModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(['src_device', 'dest_device'], [
    pytest.param(torch.device('cpu'), torch.device('cpu')),
    pytest.param(torch.device('cpu', 0), torch.device('cuda', 0)),
    pytest.param(torch.device('cuda', 0), torch.device('cpu')),
    pytest.param(torch.device('cuda', 0), torch.device('cuda', 0)),
])
def test_auto_move_data(src_device, dest_device):
    """ Test that the decorator moves the data to the device the model is on. """

    class CurrentModel(BoringModel):

        @auto_move_data
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    model = CurrentModel()
    model = model.to(dest_device)
    model.prepare_data()
    loader = model.train_dataloader()
    x = next(iter(loader))

    # test that data on source device gets moved to destination device
    x = x.to(src_device)
    assert model(x).device == dest_device, "Automoving data to same device as model failed"
