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
from pytorch_lightning.plugins import DDPSpawnPlugin
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class BoringModelDDPCPU(BoringModel):

    def on_train_start(self) -> None:
        # make sure that the model is on CPU when training
        assert self.device == torch.device("cpu")


@RunIf(skip_windows=True)
def test_ddp_cpu():
    """Tests if device is set correctely when training for DDPSpawnPlugin."""
    trainer = Trainer(num_processes=2, fast_dev_run=True)
    # assert training type plugin attributes for device setting

    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu
    assert trainer.training_type_plugin.root_device == torch.device("cpu")

    model = BoringModelDDPCPU()

    trainer.fit(model)
