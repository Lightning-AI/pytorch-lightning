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
import multiprocessing
import time
from functools import partial
from unittest.mock import Mock

import pytest
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
    existing_processes = set(multiprocessing.active_children())
    try:
        return launcher.launch(wrapped, strategy)
    finally:
        # A test timeout can interrupt joining the workers before the launcher cleans them up.
        processes = set(multiprocessing.active_children()) - existing_processes
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, pytest.fail.Exception])
def test_spawn_launch_cleans_up_after_interruption(monkeypatch, error):
    context = multiprocessing.get_context("spawn")
    existing_process = context.Process(target=time.sleep, args=(60,))
    process = context.Process(target=time.sleep, args=(60,))

    def interrupted_launch(*args, **kwargs):
        process.start()
        raise error("Interrupted launch")

    monkeypatch.setattr(_MultiProcessingLauncher, "launch", interrupted_launch)
    try:
        existing_process.start()
        with pytest.raises(error, match="Interrupted launch"):
            spawn_launch(Mock(), [torch.device("cpu"), torch.device("cpu")])
        assert not process.is_alive()
        assert existing_process.is_alive()
    finally:
        for child in (process, existing_process):
            if child.is_alive():
                child.kill()
            child.join(timeout=5)
            child.close()


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
