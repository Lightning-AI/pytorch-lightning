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
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE, _FAIRSCALE_FULLY_SHARDED_AVAILABLE
from tests.accelerators import DDPLauncher
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn import ShardedDataParallel
if _FAIRSCALE_FULLY_SHARDED_AVAILABLE:
    from fairscale.nn import FullyShardedDataParallel


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


@RunIf(min_gpus=1, skip_windows=True)
@DDPLauncher.run("--accelerator [accelerator]", max_epochs=["1"], accelerator=["ddp", "ddp_spawn"])
def test_get_model_ddp_gpu(tmpdir, args=None):
    """
    Tests that `trainer.lightning_module` extracts the model correctly when using GPU + ddp accelerators
    """

    model = TrainerGetModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        gpus=1,
        accelerator=args.accelerator
    )
    trainer.fit(model)
    return 1


@pytest.mark.parametrize(["accelerator", "wrapper"], [
    ('ddp', DistributedDataParallel),
    pytest.param(
        'ddp_sharded',
        ShardedDataParallel,
        marks=pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="FairScale not available.")
    ),
    pytest.param(
        'ddp_fully_sharded',
        FullyShardedDataParallel,
        marks=pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="FairScale not available.")
    ),
])
@RunIf(min_gpus=1, skip_windows=True)
def test_get_accelerator_wrapped_model(accelerator, wrapper, tmpdir):
    """
    Ensure we can access the wrapped accelerator model during training.
    """

    class TestModel(BoringModel):

        def on_train_start(self) -> None:
            assert isinstance(self.accelerator_model, wrapper)

        def configure_optimizers(self):
            return torch.optim.SGD(self.accelerator_model.parameters(), lr=0.1)

    model = TestModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, accelerator=accelerator, gpus=1)
    trainer.fit(model)
