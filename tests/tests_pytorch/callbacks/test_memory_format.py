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
import warnings
import torch

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import MemoryFormat
from unittest.mock import MagicMock

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from unittest.mock import MagicMock

def test_memory_format_callback_setup():
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
    callback = MemoryFormat()

    # call the setup method
    callback.setup(trainer, model)

    # check that the memory format is channels_last
    assert model.conv.weight.is_contiguous(memory_format=torch.channels_last)


def test_memory_format_callback():
    # create a mock LightningModule
    trainer = MagicMock()
    pl_module = MagicMock()

    # create a MemoryFormat callback
    memory_format_callback = MemoryFormat()

    # check that the callback sets the memory format correctly
    memory_format_callback.setup(trainer=trainer, pl_module=pl_module)
    assert pl_module.to.call_args[1]["memory_format"] == torch.channels_last

    # check that the callback resets the memory format correctly
    memory_format_callback.teardown(trainer=trainer, pl_module=pl_module)
    assert pl_module.to.call_args[1]["memory_format"] == torch.contiguous_format

    # check that the callback warns if the model doesn't have any layers benefiting from channels_last
    pl_module.modules.return_value = [torch.nn.Linear(10, 10)]
    with warnings.catch_warnings(record=True) as w:
        memory_format_callback.setup(trainer=trainer, pl_module=pl_module)
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "model does not have any layers benefiting from" in str(w[-1].message)

    # check that the callback converts input tensors to channels_last format
    memory_format_callback.convert_input = True
    batch = [torch.randn(16, 3, 32, 32), torch.randn(16, 3, 32, 32)]
    memory_format_callback.on_train_batch_start(trainer=trainer, pl_module=pl_module, batch=batch, batch_idx=0)
    for item in batch:
        assert item.is_contiguous(memory_format=torch.channels_last)