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
import os
from unittest import mock

import pytest
import torch

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import (
    DDP2Plugin,
    DDPFullyShardedPlugin,
    DDPPlugin,
    DDPShardedPlugin,
    DDPSpawnPlugin,
    DDPSpawnShardedPlugin,
    DeepSpeedPlugin,
    SingleDevicePlugin,
)
from tests.accelerators.test_dp import CustomClassificationModelDP
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf


@pytest.mark.parametrize(
    "trainer_kwargs",
    (
        pytest.param(dict(gpus=1), marks=RunIf(min_gpus=1)),
        pytest.param(dict(accelerator="dp", gpus=2), marks=RunIf(min_gpus=2)),
        pytest.param(dict(accelerator="ddp_spawn", gpus=2), marks=RunIf(min_gpus=2)),
    ),
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
    assert "ckpt" in trainer.checkpoint_callback.best_model_path

    old_weights = model.layer_0.weight.clone().detach().cpu()

    result = trainer.validate(datamodule=dm)
    assert result[0]["val_acc"] > 0.55

    result = trainer.test(datamodule=dm)
    assert result[0]["test_acc"] > 0.55

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
        plugins=CustomPlugin(device=torch.device("cpu")),
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


@pytest.mark.parametrize(
    "training_type_plugin_class",
    [
        DDPPlugin,
        DDPFullyShardedPlugin,
        DDP2Plugin,
        DDPShardedPlugin,
        DeepSpeedPlugin,
        DDPSpawnPlugin,
        DDPSpawnShardedPlugin,
    ],
)
@mock.patch.dict(
    os.environ,
    {
        "PL_TORCH_DISTRIBUTED_BACKEND": "nccl",
        "CUDA_VISIBLE_DEVICES": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.set_device")
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_set_default_nccl_environ_parameters(
    setup_distributed_mock,
    mock_set_device,
    mock_is_available,
    mock_device_count,
    training_type_plugin_class,
):
    """
    Test for setting default environ parameters.
    """
    trainer = Trainer(
        plugins=training_type_plugin_class(),
        gpus=1,
    )
    trainer.accelerator.setup_environment()
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") == "4"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"


@pytest.mark.parametrize(
    "training_type_plugin_class",
    [
        DDPPlugin,
        DDPFullyShardedPlugin,
        DDP2Plugin,
        DDPShardedPlugin,
        DeepSpeedPlugin,
        DDPSpawnPlugin,
        DDPSpawnShardedPlugin,
    ],
)
@mock.patch.dict(
    os.environ,
    {
        "NCCL_NSOCKS_PERTHREAD": "3",
        "PL_TORCH_DISTRIBUTED_BACKEND": "nccl",
        "CUDA_VISIBLE_DEVICES": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.set_device")
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_not_overriden_nccl_environ_parameters(
    setup_distributed_mock,
    mock_set_device,
    mock_is_available,
    mock_device_count,
    training_type_plugin_class,
):
    """
    Test for not setting default environ parameters when parameter is already set in `os.environ`.
    """
    trainer = Trainer(
        plugins=training_type_plugin_class(),
        gpus=1,
    )
    trainer.accelerator.setup_environment()
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") == "3"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"
