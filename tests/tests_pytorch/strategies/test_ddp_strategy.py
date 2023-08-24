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
import os
from datetime import timedelta
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

from lightning.fabric.plugins.environments import ClusterEnvironment, LightningEnvironment
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers import _SubprocessScriptLauncher
from lightning.pytorch.trainer.states import TrainerFn
from tests_pytorch.helpers.runif import RunIf


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_with_2_gpus():
    """Tests if device is set correctly when training and after teardown for DDPStrategy."""
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, DDPStrategy)
    local_rank = trainer.strategy.local_rank
    assert trainer.strategy.root_device == torch.device(f"cuda:{local_rank}")

    model = BoringModelGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory


class BarrierModel(BoringModel):
    def setup(self, stage=None):
        assert not isinstance(self.trainer.strategy.model, DistributedDataParallel)
        self.trainer.strategy.barrier("barrier before model is wrapped")

    def on_train_start(self):
        assert isinstance(self.trainer.strategy.model, DistributedDataParallel)
        self.trainer.strategy.barrier("barrier after model is wrapped")


@RunIf(min_cuda_gpus=4, standalone=True)
@mock.patch("torch.distributed.barrier")
def test_ddp_barrier_non_consecutive_device_ids(barrier_mock, tmp_path):
    """Test correct usage of barriers when device ids do not start at 0 or are not consecutive."""
    model = BoringModel()
    gpus = [1, 3]
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        accelerator="gpu",
        devices=gpus,
        strategy="ddp",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    barrier_mock.assert_any_call(device_ids=[gpus[trainer.local_rank]])


@mock.patch.dict(os.environ, {"LOCAL_RANK": "1"})
def test_incorrect_ddp_script_spawning(tmp_path):
    """Test an error message when user accidentally instructs Lightning to spawn children processes on rank > 0."""

    class WronglyImplementedEnvironment(LightningEnvironment):
        @property
        def creates_processes_externally(self):
            # returning false no matter what means Lightning would spawn also on ranks > 0 new processes
            return False

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy="ddp",
        accelerator="cpu",
        devices=2,
        plugins=[WronglyImplementedEnvironment()],
    )
    with pytest.raises(
        RuntimeError, match="Lightning attempted to launch new distributed processes with `local_rank > 0`."
    ):
        trainer.fit(model)


@RunIf(skip_windows=True)
def test_ddp_configure_ddp(mps_count_0):
    """Tests with ddp strategy."""
    model = BoringModel()
    ddp_strategy = DDPStrategy()
    trainer = Trainer(
        max_epochs=1,
        strategy=ddp_strategy,
    )
    # test wrap the model if fitting
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    assert isinstance(trainer.model, LightningModule)
    trainer.strategy.setup(trainer)
    # in DDPStrategy configure_ddp(), model wrapped by DistributedDataParallel
    assert isinstance(trainer.model, DistributedDataParallel)

    ddp_strategy = DDPStrategy()
    trainer = Trainer(
        max_epochs=1,
        strategy=ddp_strategy,
    )
    # test do not wrap the model if TrainerFn is not fitting
    trainer.state.fn = TrainerFn.VALIDATING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    trainer.strategy.setup(trainer)
    # in DDPStrategy configure_ddp(), model are still LightningModule
    assert isinstance(trainer.model, LightningModule)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("trainer_fn", [TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING])
def test_ddp_dont_configure_sync_batchnorm(trainer_fn):
    model = BoringModelGPU()
    model.layer = torch.nn.BatchNorm1d(10)
    ddp_strategy = DDPStrategy()
    trainer = Trainer(accelerator="gpu", devices=1, strategy=ddp_strategy, sync_batchnorm=True)
    trainer.state.fn = trainer_fn
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    assert isinstance(trainer.model, LightningModule)
    trainer.strategy.setup(trainer)
    # because TrainerFn is not FITTING, model is not configured with sync batchnorm
    assert not isinstance(trainer.strategy.model.layer, torch.nn.modules.batchnorm.SyncBatchNorm)


class CheckOptimizerDeviceModel(BoringModel):
    def configure_optimizers(self):
        assert all(param.device.type == "cuda" for param in self.parameters())
        super().configure_optimizers()


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("strategy", ["ddp", "ddp_spawn"])
def test_model_parameters_on_device_for_optimizer(strategy):
    """Test that the strategy has moved the parameters to the device by the time the optimizer gets created."""
    model = CheckOptimizerDeviceModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        fast_dev_run=1,
        accelerator="gpu",
        devices=1,
        strategy=strategy,
    )
    trainer.fit(model)


def test_configure_launcher_create_processes_externally():
    class MyClusterEnvironment(ClusterEnvironment):
        @property
        def creates_processes_externally(self):
            return True

        @property
        def main_address(self):
            return ""

        @property
        def main_port(self):
            return 8080

        @staticmethod
        def detect():
            return True

        def world_size(self):
            return 1

        def set_world_size(self):
            pass

        def global_rank(self):
            return 0

        def set_global_rank(self):
            pass

        def local_rank(self):
            return 0

        def node_rank(self):
            return 0

    ddp_strategy = DDPStrategy(cluster_environment=MyClusterEnvironment())
    assert ddp_strategy.launcher is None
    ddp_strategy._configure_launcher()
    assert isinstance(ddp_strategy.launcher, _SubprocessScriptLauncher)

    ddp_strategy.launcher._call_children_scripts = Mock()
    launch_fn = Mock()
    ddp_strategy.launcher.launch(launch_fn)
    ddp_strategy.launcher._call_children_scripts.assert_not_called()
    launch_fn.assert_called_once()


@mock.patch("torch.distributed.init_process_group")
def test_ddp_strategy_set_timeout(mock_init_process_group):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    model = BoringModel()
    ddp_strategy = DDPStrategy(timeout=test_timedelta)
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        strategy=ddp_strategy,
    )
    # test wrap the model if fitting
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()

    process_group_backend = trainer.strategy._get_process_group_backend()
    global_rank = trainer.strategy.cluster_environment.global_rank()
    world_size = trainer.strategy.cluster_environment.world_size()
    mock_init_process_group.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


class BoringZeroRedundancyOptimizerModel(BoringModel):
    def configure_optimizers(self):
        return ZeroRedundancyOptimizer(self.layer.parameters(), optimizer_class=torch.optim.Adam, lr=0.1)


@RunIf(min_cuda_gpus=2, skip_windows=True)
@pytest.mark.parametrize("strategy", [pytest.param("ddp", marks=RunIf(standalone=True)), "ddp_spawn"])
def test_ddp_strategy_checkpoint_zero_redundancy_optimizer(tmp_path, strategy):
    """Test to ensure that checkpoint is saved correctly when using zero redundancy optimizer."""
    model = BoringZeroRedundancyOptimizerModel()
    trainer = Trainer(accelerator="gpu", devices=2, strategy=strategy, max_steps=1)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmp_path, "model.pt")
    # need to broadcast because tmp_path is different on each process
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)


class UnusedParametersModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.intermediate_layer = torch.nn.Linear(32, 32)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.intermediate_layer(batch)
        return super().training_step(batch, batch_idx)


def test_ddp_strategy_find_unused_parameters_exception():
    """Test that the DDP strategy can change PyTorch's error message so that it's more useful for Lightning users."""
    trainer = Trainer(accelerator="cpu", devices=1, strategy="ddp", max_steps=2)
    with pytest.raises(RuntimeError, match="It looks like your LightningModule has parameters that were not used in"):
        trainer.fit(UnusedParametersModel())
