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
import torch
import torch.distributed as dist

from lightning_lite.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.strategies.launchers import _MultiProcessingLauncher
from pytorch_lightning.trainer.connectors.logger_connector.result import _Sync
from tests_pytorch.helpers.runif import RunIf


def spawn_launch(fn, parallel_devices):
    # TODO: the cluster_environment should be optional to just launch processes, but this requires lazy initialization
    strategy = DDPSpawnStrategy(parallel_devices=parallel_devices, cluster_environment=LightningEnvironment())
    launcher = _MultiProcessingLauncher(strategy=strategy)
    return launcher.launch(fn, strategy)


def result_reduce_ddp_fn(strategy):
    tensor = torch.tensor([1.0])
    sync = _Sync(strategy.reduce, _should=True, _op="SUM")
    actual = sync(tensor)
    assert actual.item() == dist.get_world_size(), "Result-Log does not work properly with DDP and Tensors"


@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    spawn_launch(result_reduce_ddp_fn, [torch.device("cpu")] * 2)
