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

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ChannelsLast


def test_channels_last_callback():
    class DummyModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    model = DummyModule()

    # create a dummy Trainer
    trainer = Trainer(max_epochs=1, devices=1)

    # create the callback
    callback = ChannelsLast()

    # call the setup method
    callback.setup(trainer, model)

    # check that the memory format is channels_last
    assert model.conv.weight.is_contiguous(memory_format=torch.channels_last)
