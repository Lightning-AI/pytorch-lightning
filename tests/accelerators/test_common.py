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

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import SingleDevicePlugin
from tests.accelerators.test_dp import CustomClassificationModelDP
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf


@pytest.mark.parametrize(
    "trainer_kwargs", (
        pytest.param(dict(gpus=1), marks=RunIf(min_gpus=1)),
        pytest.param(dict(accelerator="dp", gpus=2), marks=RunIf(min_gpus=2)),
        pytest.param(dict(accelerator="ddp_spawn", gpus=2), marks=RunIf(min_gpus=2)),
    )
)
def test_evaluate(tmpdir, trainer_kwargs):
    tutils.set_random_master_port()

    dm = ClassifDataModule()
    model = CustomClassificationModelDP()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=10,
        deterministic=True,
        **trainer_kwargs
    )

    trainer.fit(model, datamodule=dm)
    assert 'ckpt' in trainer.checkpoint_callback.best_model_path

    old_weights = model.layer_0.weight.clone().detach().cpu()

    result = trainer.validate(datamodule=dm)
    assert result[0]['val_acc'] > 0.55

    result = trainer.test(datamodule=dm)
    assert result[0]['test_acc'] > 0.55

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
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
    )
    trainer.fit(model)

    assert model.configure_sharded_model_called


class DummyModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.configure_sharded_model_called = False

    def configure_sharded_model(self):
        self.configure_sharded_model_called = True


def test_configure_sharded_model_false(tmpdir):
    """Ensure ``configure_sharded_model`` is not called, when turned off"""

    class CustomPlugin(SingleDevicePlugin):

        @property
        def call_configure_sharded_model_hook(self) -> bool:
            return False

    model = DummyModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        plugins=CustomPlugin(device=torch.device("cpu"))
    )
    trainer.fit(model)

    assert not model.configure_sharded_model_called


def test_accelerator_configure_sharded_model_called_once(tmpdir):
    """Ensure that the configure sharded model hook is called, and set to False after to ensure not called again."""

    model = DummyModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
    )
    assert trainer.accelerator.call_configure_sharded_model_hook is True
    trainer.fit(model)
    assert trainer.accelerator.call_configure_sharded_model_hook is False


def test_configure_sharded_model_called_once(tmpdir):
    """Ensure ``configure_sharded_model`` is only called once"""

    model = DummyModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
    )
    trainer.fit(model)

    assert model.configure_sharded_model_called
    model.configure_sharded_model_called = False

    assert not model.configure_sharded_model_called
