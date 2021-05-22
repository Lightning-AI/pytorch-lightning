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

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import SingleDevicePlugin
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


def test_single_cpu():
    """Tests if on_gpu and on_tpu is set correctly for single cpu plugin."""
    trainer = Trainer()
    assert isinstance(trainer.training_type_plugin, SingleDevicePlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu
    assert trainer.training_type_plugin.root_device == torch.device("cpu")


class BoringModelGPU(BoringModel):

    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device("cuda:0")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(skip_windows=True, min_gpus=1)
def test_single_gpu():
    """Tests if device is set correctely when training and after teardown for single GPU plugin."""
    trainer = Trainer(gpus=1, fast_dev_run=True)
    # assert training type plugin attributes for device setting
    assert isinstance(trainer.training_type_plugin, SingleDevicePlugin)
    assert trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu
    assert trainer.training_type_plugin.root_device == torch.device("cuda:0")

    model = BoringModelGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory
