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

import tests_pytorch.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.seed import seed_everything
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.strategies.test_dp import CustomClassificationModelDP


@pytest.mark.parametrize(
    "trainer_kwargs",
    (
        pytest.param(dict(accelerator="gpu", devices=1), marks=RunIf(min_cuda_gpus=1)),
        pytest.param(dict(strategy="dp", accelerator="gpu", devices=2), marks=RunIf(min_cuda_gpus=2)),
        pytest.param(dict(strategy="ddp_spawn", accelerator="gpu", devices=2), marks=RunIf(min_cuda_gpus=2)),
    ),
)
def test_evaluate(tmpdir, trainer_kwargs):
    tutils.set_random_main_port()
    seed_everything(1)
    dm = ClassifDataModule()
    model = CustomClassificationModelDP()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=2, limit_train_batches=10, limit_val_batches=10, **trainer_kwargs
    )

    trainer.fit(model, datamodule=dm)
    assert "ckpt" in trainer.checkpoint_callback.best_model_path

    old_weights = model.layer_0.weight.clone().detach().cpu()

    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)

    # make sure weights didn't change
    new_weights = model.layer_0.weight.clone().detach().cpu()
    torch.testing.assert_allclose(old_weights, new_weights)


def test_model_parallel_setup_called(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.configure_sharded_model_called = False
            self.layer = None

        def configure_sharded_model(self):
            self.configure_sharded_model_called = True
            self.layer = torch.nn.Linear(32, 2)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=1)
    trainer.fit(model)

    assert model.configure_sharded_model_called


@pytest.mark.parametrize(
    ["strategy", "strategy_cls"], [("DDP", DDPStrategy), ("DDP_FIND_UNUSED_PARAMETERS_FALSE", DDPStrategy)]
)
def test_strategy_str_passed_being_case_insensitive(strategy, strategy_cls):

    trainer = Trainer(strategy=strategy)
    assert isinstance(trainer.strategy, strategy_cls)
