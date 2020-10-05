import pytest
import torch
import os

from tests.metrics.test_metric import Dummy
from tests.metrics.utils import setup_ddp

torch.manual_seed(42)


def _test_ddp_sum(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": sum}
    dummy.foo = torch.tensor(1)

    dummy.sync()
    assert dummy.foo == worldsize


def _test_ddp_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = torch.tensor([1])
    dummy.sync()
    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))


def _test_ddp_sum_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.cat, "bar": sum}
    dummy.foo = torch.tensor([1])
    dummy.bar = torch.tensor(1)
    dummy.sync()
    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))
    assert dummy.bar == worldsize


@pytest.mark.parametrize("process", [_test_ddp_cat, _test_ddp_sum, _test_ddp_sum_cat])
def test_ddp(process):
    torch.multiprocessing.spawn(process, args=(2,), nprocs=2)


