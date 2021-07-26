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

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class TrainerGetModel(BoringModel):
    def on_fit_start(self):
        assert self == self.trainer.lightning_module

    def on_fit_end(self):
        assert self == self.trainer.lightning_module


def test_get_model(tmpdir):
    """
    Tests that `trainer.lightning_module` extracts the model correctly
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir, limit_train_batches=limit_train_batches, limit_val_batches=2, max_epochs=1
    )
    trainer.fit(model)


@RunIf(skip_windows=True)
def test_get_model_ddp_cpu(tmpdir):
    """
    Tests that `trainer.lightning_module` extracts the model correctly when using ddp on cpu
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        accelerator="ddp_cpu",
        num_processes=2,
    )
    trainer.fit(model)


@RunIf(min_gpus=1)
def test_get_model_gpu(tmpdir):
    """
    Tests that `trainer.lightning_module` extracts the model correctly when using GPU
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir, limit_train_batches=limit_train_batches, limit_val_batches=2, max_epochs=1, gpus=1
    )
    trainer.fit(model)
