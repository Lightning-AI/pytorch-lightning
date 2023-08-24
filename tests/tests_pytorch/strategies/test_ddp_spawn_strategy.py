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
from datetime import timedelta
from unittest import mock

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning.pytorch.trainer.states import TrainerFn
from tests_pytorch.helpers.runif import RunIf


class BoringModelDDPCPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on CPU when training
        assert self.device == torch.device("cpu")


class BoringCallbackDDPSpawnModel(BoringModel):
    def __init__(self, name: str, val: float):
        super().__init__()
        self.name = name
        self.val = val

    def validation_step(self, batch, batch_idx):
        self.log(self.name, self.val)
        return super().validation_step(batch, batch_idx)


@RunIf(skip_windows=True)
def test_ddp_cpu():
    """Tests if device is set correctly when training for DDPStrategy."""
    trainer = Trainer(devices=2, strategy="ddp_spawn", accelerator="cpu", fast_dev_run=True)
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")
    model = BoringModelDDPCPU()
    trainer.fit(model)


class CustomMultiProcessingLauncher(_MultiProcessingLauncher):
    def get_extra_results(self, trainer):
        extra = super().get_extra_results(trainer)
        extra["test_val"] = "test_val"
        return extra

    def update_main_process_results(self, trainer, extra) -> None:
        trainer.strategy.test_val = extra.pop("test_val")
        return super().update_main_process_results(trainer, extra)


class TestDDPSpawnStrategy(DDPStrategy):
    def _configure_launcher(self):
        self._launcher = CustomMultiProcessingLauncher(self)


@RunIf(skip_windows=True)
def test_ddp_spawn_add_get_queue(tmp_path):
    """Tests get_extra_results/update_main_process_results with DDPSpawnStrategy."""
    ddp_spawn_strategy = TestDDPSpawnStrategy()
    trainer = Trainer(
        default_root_dir=tmp_path, fast_dev_run=True, accelerator="cpu", devices=2, strategy=ddp_spawn_strategy
    )

    val: float = 1.0
    val_name: str = "val_acc"
    model = BoringCallbackDDPSpawnModel(val_name, val)
    dm = BoringDataModule()
    trainer.fit(model, datamodule=dm)
    assert trainer.callback_metrics[val_name] == torch.tensor(val)
    assert ddp_spawn_strategy.test_val == "test_val"


class BoringModelDDP(BoringModel):
    def on_train_start(self) -> None:
        """Check if trainer module is wrapped as DistributedDataParallel during training stage."""
        assert isinstance(self.trainer.model, DistributedDataParallel)

    def on_validation_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        if self.trainer.state.fn == TrainerFn.FITTING:
            assert isinstance(self.trainer.model, DistributedDataParallel)
        else:
            assert isinstance(self.trainer.model, LightningModule)

    def on_test_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        assert isinstance(self.trainer.model, LightningModule)

    def on_predict_start(self) -> None:
        """Check if trainer module remains as LightningModule during prediction stage."""
        assert isinstance(self.trainer.model, LightningModule)


@RunIf(skip_windows=True)
def test_ddp_spawn_configure_ddp(tmp_path):
    """Tests with ddp spawn strategy."""
    trainer = Trainer(default_root_dir=tmp_path, accelerator="cpu", devices=2, strategy="ddp_spawn", fast_dev_run=True)

    model = BoringModelDDP()

    trainer.fit(model)
    trainer.validate(model, dataloaders=model.val_dataloader())
    trainer.test(model, dataloaders=model.test_dataloader())
    trainer.predict(model, dataloaders=model.predict_dataloader())


@mock.patch("torch.distributed.init_process_group")
def test_ddp_spawn_strategy_set_timeout(mock_init_process_group):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    model = BoringModel()
    ddp_spawn_strategy = DDPStrategy(start_method="spawn", timeout=test_timedelta)
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        strategy=ddp_spawn_strategy,
    )
    # test wrap the model if fitting
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()

    process_group_backend = trainer.strategy._get_process_group_backend()
    global_rank = trainer.strategy.cluster_environment.global_rank()
    world_size = trainer.strategy.cluster_environment.world_size()
    mock_init_process_group.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


@pytest.mark.parametrize(
    ("strategy_name", "expected_ddp_kwargs"),
    [
        ("ddp_spawn", {}),
        pytest.param("ddp_fork", {}, marks=RunIf(skip_windows=True)),
        pytest.param("ddp_notebook", {}, marks=RunIf(skip_windows=True)),
        ("ddp_spawn_find_unused_parameters_false", {"find_unused_parameters": False}),
        ("ddp_spawn_find_unused_parameters_true", {"find_unused_parameters": True}),
        pytest.param(
            "ddp_fork_find_unused_parameters_false", {"find_unused_parameters": False}, marks=RunIf(skip_windows=True)
        ),
        pytest.param(
            "ddp_fork_find_unused_parameters_true", {"find_unused_parameters": True}, marks=RunIf(skip_windows=True)
        ),
        pytest.param(
            "ddp_notebook_find_unused_parameters_false",
            {"find_unused_parameters": False},
            marks=RunIf(skip_windows=True),
        ),
        pytest.param(
            "ddp_notebook_find_unused_parameters_true",
            {"find_unused_parameters": True},
            marks=RunIf(skip_windows=True),
        ),
    ],
)
def test_ddp_kwargs_from_registry(strategy_name, expected_ddp_kwargs, mps_count_0):
    trainer = Trainer(strategy=strategy_name)
    assert trainer.strategy._ddp_kwargs == expected_ddp_kwargs
