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
from torch.multiprocessing import ProcessRaisedException

import tests_pytorch.helpers.pipelines as tpipes
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer import seed_everything, Trainer
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.strategies.test_ddp_strategy import UnusedParametersModel


@RunIf(min_cuda_gpus=2, sklearn=True)
def test_multi_gpu_early_stop_ddp_spawn(tmp_path):
    seed_everything(42)

    trainer_options = {
        "default_root_dir": tmp_path,
        "callbacks": [EarlyStopping(monitor="train_acc")],
        "max_epochs": 50,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
    }

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_cuda_gpus=2)
def test_multi_gpu_model_ddp_spawn(tmp_path):
    seed_everything(42)

    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
        "enable_progress_bar": False,
    }

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model)


@RunIf(min_cuda_gpus=2)
def test_ddp_all_dataloaders_passed_to_fit(tmp_path):
    """Make sure DDP works with dataloaders passed to fit()"""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        accelerator="gpu",
        devices=[0, 1],
        strategy="ddp_spawn",
    )
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
    assert trainer.state.finished, "DDP doesn't work with dataloaders passed to fit()."


def test_ddp_spawn_find_unused_parameters_exception():
    """Test that the DDP strategy can change PyTorch's error message so that it's more useful for Lightning users."""
    trainer = Trainer(accelerator="cpu", devices=1, strategy="ddp_spawn", max_steps=2)
    with pytest.raises(
        ProcessRaisedException, match="It looks like your LightningModule has parameters that were not used in"
    ):
        trainer.fit(UnusedParametersModel())
