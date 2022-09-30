import pytest
import torch
from tests_lite.helpers.runif import RunIf

from lightning_lite.utilities.distributed import gather_all_tensors
from tests_pytorch.core.test_results import spawn_launch


def _test_all_gather_uneven_tensors(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    tensor = torch.ones(rank, device=device)
    result = gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        assert len(result[idx]) == idx
        assert (result[idx] == torch.ones_like(result[idx])).all()


def _test_all_gather_uneven_tensors_multidim(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    tensor = torch.ones(rank + 1, 2 - rank, device=device)
    result = gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        val = result[idx]
        assert val.shape == (idx + 1, 2 - idx)
        assert (val == torch.ones_like(val)).all()


@RunIf(min_torch="1.10", skip_windows=True)
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
