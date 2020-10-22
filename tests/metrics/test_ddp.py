import sys

import pytest
import torch

from tests.metrics.test_metric import Dummy
from tests.metrics.utils import setup_ddp

torch.manual_seed(42)


def _test_ddp_sum(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.sum}
    dummy.foo = torch.tensor(1)

    dummy._sync_dist()
    assert dummy.foo == worldsize


def _test_ddp_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [torch.tensor([1])]
    dummy._sync_dist()
    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))


def _test_ddp_sum_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.cat, "bar": torch.sum}
    dummy.foo = [torch.tensor([1])]
    dummy.bar = torch.tensor(1)
    dummy._sync_dist()
    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))
    assert dummy.bar == worldsize


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("process", [_test_ddp_cat, _test_ddp_sum, _test_ddp_sum_cat])
def test_ddp(process):
    torch.multiprocessing.spawn(process, args=(2,), nprocs=2)
