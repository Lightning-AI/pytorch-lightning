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
from functools import partial

import torch
import torch.distributed as dist
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher
from lightning.pytorch.trainer.connectors.logger_connector.result import _Sync

from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.models.test_tpu import wrap_launch_function


def spawn_launch(fn, parallel_devices):
    # TODO: the accelerator and cluster_environment should be optional to just launch processes, but this requires lazy
    # initialization to be implemented
    device_to_accelerator = {"cuda": CUDAAccelerator, "mps": MPSAccelerator, "cpu": CPUAccelerator}
    accelerator_cls = device_to_accelerator[parallel_devices[0].type]
    strategy = DDPStrategy(
        accelerator=accelerator_cls(),
        parallel_devices=parallel_devices,
        cluster_environment=LightningEnvironment(),
        start_method="spawn",
    )
    launcher = _MultiProcessingLauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def result_reduce_ddp_fn(strategy):
    tensor = torch.tensor([1.0])
    sync = _Sync(strategy.reduce, _should=True, _op="SUM")
    actual = sync(tensor)
    assert actual.item() == dist.get_world_size()


@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    spawn_launch(result_reduce_ddp_fn, [torch.device("cpu")] * 2)
