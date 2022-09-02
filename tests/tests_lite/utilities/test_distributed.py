import os

import pytest
import torch
from tests_lite.helpers.runif import RunIf
from torch import multiprocessing as mp

from lightning_lite.utilities.distributed import gather_all_tensors


def _test_all_gather_uneven_tensors(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"

    if backend == "nccl":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # initialize the process group
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

    tensor = torch.ones(rank, device=device)
    result = gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        assert len(result[idx]) == idx
        assert (result[idx] == torch.ones_like(result[idx])).all()


def _test_all_gather_uneven_tensors_multidim(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"

    if backend == "nccl":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # initialize the process group
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
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
@pytest.mark.parametrize("backend", [pytest.param("nccl", marks=RunIf(min_cuda_gpus=2)), "gloo"])
def test_gather_all_tensors(backend, process):
    tutils.set_random_main_port()
    mp.spawn(process, args=(2, backend), nprocs=2)
