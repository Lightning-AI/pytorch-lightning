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
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

from tests_pytorch.helpers.runif import RunIf


class TrainerGetModel(BoringModel):
    def on_fit_start(self):
        assert self == self.trainer.lightning_module

    def on_fit_end(self):
        assert self == self.trainer.lightning_module


def test_get_model(tmp_path):
    """Tests that `trainer.lightning_module` extracts the model correctly."""
    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmp_path, limit_train_batches=limit_train_batches, limit_val_batches=2, max_epochs=1
    )
    trainer.fit(model)


@RunIf(skip_windows=True)
def test_get_model_ddp_cpu(tmp_path):
    """Tests that `trainer.lightning_module` extracts the model correctly when using ddp on cpu."""
    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("gpu", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
def test_get_model_gpu(tmp_path, accelerator):
    """Tests that `trainer.lightning_module` extracts the model correctly when using GPU."""
    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        accelerator=accelerator,
        devices=1,
    )
    trainer.fit(model)
