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
import pickle

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.strategies import SingleDeviceStrategy
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


def test_single_cpu():
    """Tests if device is set correctly for single CPU strategy."""
    trainer = Trainer()
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device("cuda:0")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(skip_windows=True, min_gpus=1)
def test_single_gpu():
    """Tests if device is set correctly when training and after teardown for single GPU strategy."""
    trainer = Trainer(accelerator="gpu", devices=1, fast_dev_run=True)
    # assert training strategy attributes for device setting
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("cuda:0")

    model = BoringModelGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory


class MockOptimizer:
    ...


def test_strategy_pickle():
    strategy = SingleDeviceStrategy("cpu")
    optimizer = MockOptimizer()

    strategy.optimizers = [optimizer]
    assert isinstance(strategy.optimizers[0], MockOptimizer)
    assert isinstance(strategy._lightning_optimizers[0], LightningOptimizer)

    state = pickle.dumps(strategy)
    # dumping did not get rid of the lightning optimizers
    assert isinstance(strategy._lightning_optimizers[0], LightningOptimizer)
    strategy_reloaded = pickle.loads(state)
    # loading restores the lightning optimizers
    assert isinstance(strategy_reloaded._lightning_optimizers[0], LightningOptimizer)
