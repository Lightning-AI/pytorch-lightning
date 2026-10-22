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
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection, _Sync
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


def result_reduce_epoch_mean_ddp_fn(sync_dist_op, expected, strategy):
    result = _ResultCollection(training=True)
    values = (10.0, 14.0) if strategy.global_rank == 0 else (2.0, 4.0)
    batch_sizes = (1, 1) if strategy.global_rank == 0 else (1, 3)
    for value, batch_size in zip(values, batch_sizes):
        result.log(
            "training_step",
            "mean",
            torch.tensor(value),
            reduce_fx="mean",
            sync_dist=True,
            sync_dist_fn=strategy.reduce,
            sync_dist_op=sync_dist_op,
            batch_size=batch_size,
        )
    # The local means are 12 and 3.5. Without an explicit operation, retain
    # the global sample-weighted mean (24 + 14) / (2 + 4).
    torch.testing.assert_close(result["training_step.mean"].compute(), torch.tensor(expected))


@pytest.mark.parametrize(
    ("sync_dist_op", "expected"),
    [
        (None, 38 / 6),
        ("sum", 15.5),
        ("mean", 7.75),
        ("avg", 7.75),
        ("max", 12.0),
        ("min", 3.5),
        (dist.ReduceOp.SUM, 15.5),
        (dist.ReduceOp.MAX, 12.0),
        (dist.ReduceOp.MIN, 3.5),
    ],
)
@RunIf(skip_windows=True)
def test_result_reduce_epoch_mean_ddp(sync_dist_op, expected):
    fn = partial(result_reduce_epoch_mean_ddp_fn, sync_dist_op, expected)
    spawn_launch(fn, [torch.device("cpu")] * 2)
