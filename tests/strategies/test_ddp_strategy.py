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
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(skip_windows=True, min_gpus=2, standalone=True)
def test_ddp_with_2_gpus():
    """Tests if device is set correctly when training and after teardown for DDPStrategy."""
    trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp", fast_dev_run=True)
    # assert training type plugin attributes for device setting
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


@RunIf(min_gpus=4, standalone=True)
@mock.patch("torch.distributed.barrier")
def test_ddp_barrier_non_consecutive_device_ids(barrier_mock, tmpdir):
    """Test correct usage of barriers when device ids do not start at 0 or are not consecutive."""
    model = BoringModel()
    gpus = [1, 3]
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, gpus=gpus, strategy="ddp")
    trainer.fit(model)
    barrier_mock.assert_any_call(device_ids=[gpus[trainer.local_rank]])


@mock.patch.dict(os.environ, {"LOCAL_RANK": "1"})
def test_incorrect_ddp_script_spawning(tmpdir):
    """Test an error message when user accidentally instructs Lightning to spawn children processes on rank > 0."""

    class WronglyImplementedEnvironment(LightningEnvironment):
        @property
        def creates_processes_externally(self):
            # returning false no matter what means Lightning would spawn also on ranks > 0 new processes
            return False

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy="ddp",
        num_processes=2,
        plugins=[WronglyImplementedEnvironment()],
    )
    with pytest.raises(
        RuntimeError, match="Lightning attempted to launch new distributed processes with `local_rank > 0`."
    ):
        trainer.fit(model)


@RunIf(skip_windows=True)
def test_ddp_configure_ddp():
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


@RunIf(min_gpus=1)
@pytest.mark.parametrize(
    "trainer_fn", (TrainerFn.VALIDATING, TrainerFn.TUNING, TrainerFn.TESTING, TrainerFn.PREDICTING)
)
def test_ddp_dont_configure_sync_batchnorm(trainer_fn):
    model = BoringModelGPU()
    model.layer = torch.nn.BatchNorm1d(10)
    ddp_strategy = DDPStrategy()
    trainer = Trainer(gpus=1, strategy=ddp_strategy, sync_batchnorm=True)
    trainer.state.fn = trainer_fn
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    assert isinstance(trainer.model, LightningModule)
    trainer.strategy.setup(trainer)
    # because TrainerFn is not FITTING, model is not configured with sync batchnorm
    assert not isinstance(trainer.strategy.model.layer, torch.nn.modules.batchnorm.SyncBatchNorm)
