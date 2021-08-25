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
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import DDPForkPlugin
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


@RunIf(min_gpus=2)
def test_ddp_fork_plugin_shared_data(tmpdir):
    class ValidateSharedDataCallback(Callback):
        def __init__(self, dataset):
            self.data_reference = dataset.data_ptr()

        def on_train_start(self, trainer, pl_module):
            assert trainer.train_dataloder.dataset.data.data_ptr() == self.data_reference

    dataset = RandomDataset(32, 64)
    dataloader = DataLoader(dataset)

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir, accelerator="ddp_fork", gpus=2, callbacks=ValidateSharedDataCallback(dataset)
    )
    trainer.fit(model, train_dataloder=dataloader)


@RunIf(min_gpus=2)
def test_ddp_fork_plugin(tmpdir):

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=2, accelerator="ddp_fork")

    assert isinstance(trainer.training_type_plugin, DDPForkPlugin)
    assert trainer.training_type_plugin.on_gpu
