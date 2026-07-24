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

import pytest
import torch
import torch.distributed as dist

from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher
from lightning.pytorch.trainer.connectors.logger_connector.result import _Metadata, _ResultMetric, _Sync
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


# flaky with "torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGABRT"
@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    spawn_launch(result_reduce_ddp_fn, [torch.device("cpu")] * 2)


def result_metric_mean_uneven_batches_fn(strategy):
    # Ranks log a different number of batches (3 vs 2), each with batch_size=1. The epoch value must
    # be the exact batch-size-weighted mean over all 5 updates, (1 + 2 + 3 + 10 + 20) / 5 = 7.2.
    # Mean-syncing the int64 batch-size accumulator floors the denominator (5 / 2 ranks -> 2 instead
    # of 2.5), which would inflate the result to (36 / 2) / 2 = 9.0.
    metadata = _Metadata("validation_step", "val_metric", on_step=False, on_epoch=True)
    metadata.sync = _Sync(strategy.reduce, _should=True)
    metric = _ResultMetric(metadata, is_tensor=True)
    values = ([1.0, 2.0, 3.0], [10.0, 20.0])[dist.get_rank()]
    for value in values:
        metric.update(torch.tensor(value), batch_size=1)
    assert metric.compute().item() == pytest.approx(7.2)


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_result_metric_mean_with_uneven_batch_counts():
    """Mean-reduced epoch metrics must not floor the cross-rank batch-count denominator."""
    spawn_launch(result_metric_mean_uneven_batches_fn, [torch.device("cpu")] * 2)
