import os
import pytest
import torch
import torch.nn as nn

from pytorch_lightning.utilities import AllGatherGrad


def setup_ddp(rank, world_size):
    """ Setup ddp enviroment """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8088"

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _test_all_gather(rank, world_size):
    setup_ddp(rank, world_size)

    tensor1 = torch.randn(8)
    tensor2 = torch.randn(8, 16, 32)

    tensor1_gathered = AllGatherGrad.apply(tensor1)
    tensor2_gathered = AllGatherGrad.apply(tensor2)

    assert torch.sum(tensor1_gathered[rank] - tensor1) == 0
    assert torch.sum(tensor2_gathered[rank] - tensor2) == 0


def test_all_gather():
    world_size = 3
    torch.multiprocessing.spawn(_test_all_gather, args=(world_size,), nprocs=world_size)
