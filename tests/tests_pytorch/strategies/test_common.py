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
import torch

from lightning.pytorch import Trainer
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@pytest.mark.parametrize(
    "trainer_kwargs",
    [
        pytest.param({"accelerator": "gpu", "devices": 1}, marks=RunIf(min_cuda_gpus=1)),
        pytest.param({"strategy": "ddp_spawn", "accelerator": "gpu", "devices": 2}, marks=RunIf(min_cuda_gpus=2)),
        pytest.param({"accelerator": "mps", "devices": 1}, marks=RunIf(mps=True)),
    ],
)
@RunIf(sklearn=True)
def test_evaluate(tmpdir, trainer_kwargs):
    dm = ClassifDataModule()
    model = ClassificationModel()
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
    torch.testing.assert_close(old_weights, new_weights)
