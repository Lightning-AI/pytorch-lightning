import os
import pytest
import sys
import torch
import torch.nn as nn

from pytorch_lightning.utilities import AllGatherGrad


def setup_ddp(rank, world_size):
    """ Setup ddp enviroment """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8088"

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _test_all_gather_ddp(rank, world_size):
    setup_ddp(rank, world_size)

    tensor1 = torch.ones(8, requires_grad=True)
    tensor2 = torch.ones((8, 16, 32), requires_grad=True)

    tensor1_gathered = AllGatherGrad.apply(tensor1)
    tensor2_gathered = AllGatherGrad.apply(tensor2)

    tensor1_gathered = tensor1_gathered * rank
    tensor2_gathered = tensor2_gathered * rank

    tensor1_gathered.sum().backward()
    tensor2_gathered.sum().backward()

    grad1 = torch.zeros_like(tensor1.grad).fill_(torch.arange(world_size).sum().float())
    grad2 = torch.zeros_like(tensor2.grad).fill_(torch.arange(world_size).sum().float())

    assert torch.allclose(grad1, tensor1.grad)
    assert torch.allclose(grad2, tensor2.grad)


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_all_gather_ddp():
    world_size = 3
    torch.multiprocessing.spawn(_test_all_gather_ddp, args=(world_size,), nprocs=world_size)
