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
from torch.nn.parallel.distributed import DistributedDataParallel

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel


class BoringModelDDP(BoringModel):
    def on_train_start(self) -> None:
        """Check if trainer module is wrapped as DistributedDataParallel during training stage."""
        assert isinstance(self.trainer.model, DistributedDataParallel)

    def on_test_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        assert isinstance(self.trainer.model, LightningModule)

    def on_predict_start(self) -> None:
        """Check if trainer module remains as LightningModule during prediction stage."""
        assert isinstance(self.trainer.model, LightningModule)


class BoringModelSharded(BoringModel):
    def on_train_start(self) -> None:
        """Check if trainer module is wrapped as ShardedDataParallel during training stage."""
        assert isinstance(self.trainer.model, ShardedDataParallel)

    def on_test_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        assert isinstance(self.trainer.model, LightningModule)

    def on_predict_start(self) -> None:
        """Check if trainer module remains as LightningModule during prediction stage."""
        assert isinstance(self.trainer.model, LightningModule)


@RunIf(skip_windows=True)
@pytest.mark.parametrize(
    ["accelerator"],
    [
        ("ddp",),
        ("ddp_spawn",),
    ],
)
def test_ddp_cpu(accelerator):
    """Tests with ddp or ddp_spawn plugin."""
    trainer = Trainer(num_processes=2, accelerator=accelerator, fast_dev_run=True)

    model = BoringModelDDP()

    trainer.fit(model)


@RunIf(fairscale=True)
@pytest.mark.parametrize(["accelerator"], [("ddp_sharded",), ("ddp_sharded_spawn",)])
def test_sharded_cpu(accelerator):
    """Tests with ddp_sharded or ddp_sharded_spawn plugin."""
    trainer = Trainer(accelerator=accelerator, fast_dev_run=True)

    model = BoringModelSharded()

    trainer.fit(model)
