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

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class TrainerGetModel(BoringModel):

    def on_fit_start(self):
        assert self == self.trainer.lightning_module

    def on_fit_end(self):
        assert self == self.trainer.lightning_module


def test_get_model(tmpdir):
    """
    Tests that `trainer.lightning_module` extracts the model correctly
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
    )
    trainer.fit(model)


@RunIf(skip_windows=True)
def test_get_model_ddp_cpu(tmpdir):
    """
    Tests that `trainer.lightning_module` extracts the model correctly when using ddp on cpu
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        accelerator='ddp_cpu',
        num_processes=2,
    )
    trainer.fit(model)


@RunIf(min_gpus=1)
def test_get_model_gpu(tmpdir):
    """
    Tests that `trainer.lightning_module` extracts the model correctly when using GPU
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        gpus=1,
    )
    trainer.fit(model)


class TestShardedModule(BoringModel):

    def __init__(self, accelerator: str):
        super().__init__()
        self.accelerator = accelerator

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.sharded_module.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        if self.accelerator == 'ddp_spawn':
            from torch.nn.parallel import DistributedDataParallel
            assert isinstance(self.sharded_module, DistributedDataParallel)
            assert isinstance(self.trainer.sharded_module, DistributedDataParallel)
        else:
            assert self.sharded_module == self
            assert self.trainer.sharded_module == self


@pytest.mark.parametrize(["accelerator", "num_processes"],
                         [(None, 1), pytest.param('ddp_spawn', 2, marks=RunIf(skip_windows=True))])
def test_get_sharded_module(tmpdir, accelerator, num_processes):
    """
    Tests that `trainer.sharded_module` and `model.sharded_module` extracts the accelerator model.
    """

    model = TestShardedModule(accelerator)

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=accelerator,
        num_processes=num_processes,
        fast_dev_run=True,
    )
    trainer.fit(model)
