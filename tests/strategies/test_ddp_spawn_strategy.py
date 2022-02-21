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
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.strategies.launchers.spawn import _SpawnLauncher
from pytorch_lightning.trainer.states import TrainerFn
from tests.helpers.boring_model import BoringDataModule, BoringModel
from tests.helpers.runif import RunIf


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

    def add_to_queue(self, queue) -> None:
        queue.put("test_val")
        return super().add_to_queue(queue)

    def get_from_queue(self, queue) -> None:
        self.test_val = queue.get()
        return super().get_from_queue(queue)


@RunIf(skip_windows=True, skip_49370=True)
def test_ddp_cpu():
    """Tests if device is set correctly when training for DDPSpawnStrategy."""
    trainer = Trainer(devices=2, accelerator="cpu", fast_dev_run=True)
    # assert training type plugin attributes for device setting

    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")

    model = BoringModelDDPCPU()

    trainer.fit(model)


@RunIf(min_gpus=2)
def test_ddp_spawn_extra_parameters(tmpdir):
    """Tests if device is set correctly when training for DDPSpawnStrategy and tests add_to_queue/get_from_queue
    with Lightning Module (deprecated way)."""
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=2, strategy="ddp_spawn")

    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert trainer.strategy.root_device == torch.device("cuda:0")

    val: float = 1.0
    val_name: str = "val_acc"
    model = BoringCallbackDDPSpawnModel(val_name, val)
    dm = BoringDataModule()
    trainer.fit(model, datamodule=dm)
    assert trainer.callback_metrics[val_name] == torch.tensor(val)
    assert model.test_val == "test_val"


class CustomSpawnLauncher(_SpawnLauncher):
    def add_to_queue(self, trainer, queue) -> None:
        queue.put("new_test_val")
        return super().add_to_queue(trainer, queue)

    def get_from_queue(self, trainer: Trainer, queue) -> None:
        trainer.strategy.new_test_val = queue.get()
        return super().get_from_queue(trainer, queue)


class TestDDPSpawnStrategy(DDPSpawnStrategy):
    def _configure_launcher(self):
        self._launcher = CustomSpawnLauncher(self)


@RunIf(skip_windows=True, skip_49370=True)
def test_ddp_spawn_add_get_queue(tmpdir):
    """Tests add_to_queue/get_from_queue with DDPSpawnStrategy."""

    ddp_spawn_strategy = TestDDPSpawnStrategy()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, num_processes=2, strategy=ddp_spawn_strategy)

    val: float = 1.0
    val_name: str = "val_acc"
    model = BoringCallbackDDPSpawnModel(val_name, val)
    dm = BoringDataModule()
    trainer.fit(model, datamodule=dm)
    assert trainer.callback_metrics[val_name] == torch.tensor(val)
    assert ddp_spawn_strategy.new_test_val == "new_test_val"


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


@RunIf(skip_windows=True, skip_49370=True, skip_hanging_spawn=True)
def test_ddp_spawn_configure_ddp(tmpdir):
    """Tests with ddp spawn strategy."""
    trainer = Trainer(default_root_dir=tmpdir, num_processes=2, strategy="ddp_spawn", fast_dev_run=True)

    model = BoringModelDDP()

    trainer.fit(model)
    trainer.validate(model, dataloaders=model.val_dataloader())
    trainer.test(model, dataloaders=model.test_dataloader())
    trainer.predict(model, dataloaders=model.predict_dataloader())


@pytest.mark.parametrize("trainer_fn", [TrainerFn.FITTING, "other"])
def test_ddp_spawn_transfer_weights(tmpdir, trainer_fn):
    """Tests that the spawn strategy transfers the new weights to the main process and deletes the temporary
    file."""
    model = Mock(wraps=BoringModel(), spec=BoringModel)
    strategy = DDPSpawnStrategy()
    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy)
    trainer.strategy.connect(model)
    trainer.state.fn = trainer_fn  # pretend we are in a particular trainer state
    temp_file = Path(tmpdir, ".temp.ckpt")

    assert not temp_file.exists()
    spawn_output = strategy._launcher._collect_rank_zero_results(trainer, {})

    model.state_dict.assert_called_once()
    if trainer_fn == TrainerFn.FITTING:
        assert spawn_output.weights_path == str(temp_file)
        assert temp_file.exists()
    else:
        assert spawn_output.weights_path is None
        assert not temp_file.exists()

    # <-- here would normally be the multiprocessing boundary
    strategy._launcher._recover_results_in_main_process(spawn_output, trainer)
    assert model.load_state_dict.call_count == int(spawn_output.weights_path is not None)
    assert not temp_file.exists()
