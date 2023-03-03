from functools import partial

import pytest
import torch

from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning.fabric.utilities.distributed import _gather_all_tensors
from tests_fabric.helpers.runif import RunIf


def wrap_launch_function(fn, strategy, *args, **kwargs):
    # the launcher does not manage this automatically. explanation available in:
    # https://github.com/Lightning-AI/lightning/pull/14926#discussion_r982976718
    strategy.setup_environment()
    return fn(*args, **kwargs)


def spawn_launch(fn, parallel_devices):
    """Copied from ``tests_pytorch.core.test_results.spawn_launch``"""
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


def _test_all_gather_uneven_tensors(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    tensor = torch.ones(rank, device=device)
    result = _gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        assert len(result[idx]) == idx
        assert (result[idx] == torch.ones_like(result[idx])).all()


def _test_all_gather_uneven_tensors_multidim(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    tensor = torch.ones(rank + 1, 2 - rank, device=device)
    result = _gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        val = result[idx]
        assert val.shape == (idx + 1, 2 - idx)
        assert (val == torch.ones_like(val)).all()


@RunIf(skip_windows=True)
@pytest.mark.parametrize(
    "process",
    [
        _test_all_gather_uneven_tensors_multidim,
        _test_all_gather_uneven_tensors,
    ],
)
@pytest.mark.parametrize(
    "devices",
    [
        pytest.param([torch.device("cuda:0"), torch.device("cuda:1")], marks=RunIf(min_cuda_gpus=2)),
        [torch.device("cpu")] * 2,
    ],
)
def test_gather_all_tensors(devices, process):
    spawn_launch(process, devices)
