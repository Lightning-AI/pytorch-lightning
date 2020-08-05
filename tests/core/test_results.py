import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult
import tests.base.develop_utils as tutils
import sys


def _setup_ddp(rank, worldsize):
    import os

    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _ddp_test_fn(rank, worldsize, result_cls: Result):
    _setup_ddp(rank, worldsize)
    tensor = torch.tensor([1.0])

    res = result_cls.log("test_tensor", tensor, sync_ddp=True, sync_ddp_op=torch.distributed.ReduceOp.SUM)

    assert res["test_tensor"].item() == dist.get_world_size(), "Result-Log does not work properly with DDP and Tensors"


@pytest.mark.parametrize("result_cls", [Result, TrainResult, EvalResult])
@pytest.mark.skipif(sys.platform == "win32" , reason="DDP not available on windows")
def test_result_reduce_ddp(result_cls):
    """Make sure result logging works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize, result_cls), nprocs=worldsize)
